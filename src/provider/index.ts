import vscode from 'vscode';
import { AuthManager } from '../auth';
import { DeepSeekClient } from '../client';
import { getApiModelId, getBaseUrl, getMaxTokens } from '../config';
import {
	API_KEY_REQUIRED_DETAIL,
	COMMAND_PREFIX,
	MODELS,
	PROVIDER_VENDOR,
	THINKING_EFFORT_CONFIGURATION_SCHEMA,
	TOKEN_CALIBRATION_KEY,
} from '../consts';
import { logger } from '../logger';
import type { TokenUsageTracker } from '../tokenUsage';
import type { DeepSeekTool, DeepSeekToolCall, DeepSeekUsage, ModelDefinition } from '../types';
import { type ReasoningEntry, pruneReasoningCache } from './cache';
import { convertMessages, convertTools, countRequestChars } from './convert';
import {
	createVisionModelGetter,
	getConfiguredVisionModelId,
	resolveImageMessages,
	setVisionProxyModel,
} from './vision';

/**
 * NOTE: Non-public API surface.
 *
 * The fields below (`configurationSchema` on chat info, `modelConfiguration`
 * on response options, plus `isUserSelectable` / `statusIcon`) are not part
 * of the stable `vscode.LanguageModelChat*` typings yet. They are the same
 * shape currently consumed by GitHub Copilot Chat to render a per-model
 * config dropdown in the model picker (see Copilot Chat's built-in
 * providers, e.g. its OpenAI/Anthropic providers using `reasoningEffort`).
 *
 * If/when VS Code stabilizes these as proposed API, switch to the official
 * types and drop the casts below.
 */

type ThinkingEffort = 'none' | 'high' | 'max';

/**
 * Non-public: Copilot Chat passes the user's per-model picker selections
 * back to providers via `modelConfiguration` (newer hosts) / `configuration`
 * (older hosts) on the response options. Both names are checked at runtime.
 */
type ModelConfigurationOptions = vscode.ProvideLanguageModelChatResponseOptions & {
	readonly modelConfiguration?: Record<string, unknown>;
	readonly configuration?: Record<string, unknown>;
};

/**
 * Non-public: extra fields on `LanguageModelChatInformation` consumed by the
 * Copilot Chat model picker — `isUserSelectable` controls picker visibility,
 * `statusIcon` renders a leading icon (e.g. warning when key missing), and
 * `configurationSchema` declares the per-model dropdown schema.
 */
type ModelPickerChatInformation = vscode.LanguageModelChatInformation & {
	readonly isUserSelectable: boolean;
	readonly statusIcon?: vscode.ThemeIcon;
	readonly configurationSchema?: typeof THINKING_EFFORT_CONFIGURATION_SCHEMA;
};

type TokenCalibration = {
	readonly charsPerToken: number;
	readonly samples: number;
	readonly updatedAt: number;
};

/**
 * DeepSeek Chat Provider — implements vscode.LanguageModelChatProvider so
 * DeepSeek V4 models appear directly in the Copilot Chat model picker.
 */
export class DeepSeekChatProvider implements vscode.LanguageModelChatProvider {
	private readonly authManager: AuthManager;
	private readonly globalState: vscode.Memento;
	private readonly onDidChangeLanguageModelChatInformationEmitter = new vscode.EventEmitter<void>();
	private isActive = true;

	readonly onDidChangeLanguageModelChatInformation =
		this.onDidChangeLanguageModelChatInformationEmitter.event;

	/** reasoning text → tool_call IDs cache. */
	private readonly reasoningCache = new Map<string, ReasoningEntry>();

	/**
	 * Fingerprint of the first user message seen in the current conversation.
	 * Used to detect genuine conversation resets without false-positives from
	 * agent mode where messages.length can be small (e.g. 2) mid-task.
	 */
	private lastConversationId: string | undefined;

	/** Rolling per-turn cache miss rates for anomaly detection (last 5 turns). */
	private recentCacheMissRates: number[] = [];

	/** Vision proxy: resolver + cached model. */
	private readonly vision = createVisionModelGetter();

	/**
	 * Adaptive chars-per-token ratio, calibrated from actual usage data.
	 * Updated via exponential moving average each time the API reports real token counts.
	 */
	private charsPerToken = 3.0;
	private tokenCalibrationSamples = 0;

	/** Session-scoped token usage tracker. */
	private readonly tokenUsageTracker: TokenUsageTracker;

	constructor(context: vscode.ExtensionContext, tokenUsageTracker: TokenUsageTracker) {
		this.authManager = new AuthManager(context);
		this.tokenUsageTracker = tokenUsageTracker;
		this.globalState = context.globalState;
		this.restoreTokenCalibration();

		context.subscriptions.push(
			this.onDidChangeLanguageModelChatInformationEmitter,
			// Settings-based fallback API key + vision model changes.
			vscode.workspace.onDidChangeConfiguration((e) => {
				if (e.affectsConfiguration(`${COMMAND_PREFIX}.apiKey`)) {
					this.onDidChangeLanguageModelChatInformationEmitter.fire();
				}

				if (e.affectsConfiguration(`${COMMAND_PREFIX}.visionModel`)) {
					this.vision.reset();
				}
			}),
			// Multi-window: SecretStorage changes don't fire onDidChangeConfiguration.
			// When another window sets/clears the API key, refresh this window's
			// model picker so the warning state stays in sync.
			context.secrets.onDidChange((e) => {
				if (e.key === `${COMMAND_PREFIX}.apiKey`) {
					this.onDidChangeLanguageModelChatInformationEmitter.fire();
				}
			}),
		);
	}

	// ---- Public commands ----

	async configureApiKey(): Promise<void> {
		const saved = await this.authManager.promptForApiKey();
		if (saved) {
			this.onDidChangeLanguageModelChatInformationEmitter.fire();
		}
	}

	async clearApiKey(): Promise<void> {
		this.tokenUsageTracker.reset();
		await this.authManager.deleteApiKey();
		this.onDidChangeLanguageModelChatInformationEmitter.fire();
		vscode.window.showInformationMessage('DeepSeek API key removed.');
	}

	async manage(): Promise<void> {
		const hasKey = await this.hasApiKey();
		const visionId = getConfiguredVisionModelId();
		const items: Array<vscode.QuickPickItem & { run: () => Promise<void> | void }> = [
			{
				label: '$(key) Set DeepSeek API Key',
				description: hasKey ? 'replace saved key' : 'required before using DeepSeek V4',
				run: () => this.configureApiKey(),
			},
			...(hasKey
				? [
						{
							label: '$(trash) Clear DeepSeek API Key',
							description: 'remove key from VS Code SecretStorage',
							run: () => this.clearApiKey(),
						},
					]
				: []),
			{
				label: '$(eye) Set Vision Proxy Model',
				description: visionId || 'auto-detect',
				detail: 'Used only to describe image attachments before DeepSeek V4 receives the prompt.',
				run: () => this.setVisionProxyModel(),
			},
			{
				label: '$(settings-gear) Open DeepSeek Settings',
				run: () =>
					vscode.commands.executeCommand('workbench.action.openSettings', COMMAND_PREFIX),
			},
			{
				label: '$(output) Show DeepSeek Logs',
				run: () => logger.show(),
			},
			{
				label: '$(link-external) Open DeepSeek API Keys',
				run: () =>
					vscode.env.openExternal(vscode.Uri.parse('https://platform.deepseek.com/api_keys')),
			},
		];

		const picked = await vscode.window.showQuickPick(items, {
			title: 'DeepSeek V4 Bridge for Copilot Chat',
			placeHolder: hasKey
				? 'Manage DeepSeek V4 Bridge inside Copilot Chat'
				: 'Set an API key to enable DeepSeek V4 Bridge in Copilot Chat',
			matchOnDescription: true,
			matchOnDetail: true,
		});

		await picked?.run();
	}

	async hasApiKey(): Promise<boolean> {
		return this.authManager.hasApiKey();
	}

	/** Force Copilot Chat to re-query model information (including configurationSchema). */
	refreshModelPicker(): void {
		this.onDidChangeLanguageModelChatInformationEmitter.fire();
	}

	async prepareForDeactivate(): Promise<void> {
		this.isActive = false;
		this.onDidChangeLanguageModelChatInformationEmitter.fire();

		// Force the host to re-pull `provideLanguageModelChatInformation` synchronously
		// before the extension unloads. With `isActive = false` we now return [],
		// which makes Copilot Chat drop DeepSeek models from the picker immediately
		// instead of leaving stale entries behind after deactivate. The returned
		// model list itself is unused — we only call this for its side effect.
		try {
			await vscode.lm.selectChatModels({ vendor: PROVIDER_VENDOR });
		} catch (error) {
			logger.warn('Failed to refresh DeepSeek models during deactivate', error);
		}
	}

	/** See provider/vision.ts */
	async setVisionProxyModel(): Promise<void> {
		await setVisionProxyModel();
	}

	// ---- LanguageModelChatProvider ----

	async provideLanguageModelChatInformation(
		_options: vscode.PrepareLanguageModelChatModelOptions,
		_token: vscode.CancellationToken,
	): Promise<vscode.LanguageModelChatInformation[]> {
		if (!this.isActive) {
			return [];
		}

		const hasKey = await this.authManager.hasApiKey();
		return MODELS.map((model) => toChatInfo(model, hasKey));
	}

	async provideLanguageModelChatResponse(
		modelInfo: vscode.LanguageModelChatInformation,
		messages: readonly vscode.LanguageModelChatRequestMessage[],
		options: vscode.ProvideLanguageModelChatResponseOptions,
		progress: vscode.Progress<vscode.LanguageModelResponsePart>,
		token: vscode.CancellationToken,
	): Promise<void> {
		// Set active model for cost calculation before proceeding.
		this.tokenUsageTracker.setModel(modelInfo.id);

		const modelDef = MODELS.find((m) => m.id === modelInfo.id);
		if (!modelDef) {
			throw vscode.LanguageModelError.NotFound(`Unknown DeepSeek V4 model: ${modelInfo.id}`);
		}

		const isThinkingModel = modelDef?.capabilities.thinking ?? false;
		const thinkingEffort = getConfiguredThinkingEffort(options as ModelConfigurationOptions);
		const maxTokens = getMaxTokens();

		// Log Model Turn for agent debug visibility.
		logger.info(
			`Model Turn: ${modelDef.name}` +
				` | thinking=${thinkingEffort}` +
				` | messages=${messages.length}` +
				` | tools=${options.tools?.length ?? 0}` +
				` | maxTokens=${maxTokens ?? 'unlimited'}`,
		);

		const apiKey = await this.authManager.getApiKey();
		if (!apiKey) {
			await this.showApiKeyRequiredPrompt();
			throw vscode.LanguageModelError.NoPermissions(
				'DeepSeek V4 Bridge API key is not configured. Run "DeepSeek V4 Bridge: Set API Key".',
			);
		}

		const baseUrl = getBaseUrl();
		const client = new DeepSeekClient(baseUrl, apiKey);

		// Detect a genuine new conversation by comparing the first user message
		// fingerprint. The old messages.length<=2 heuristic caused false positives
		// in agent mode (where a fresh tool-call loop still starts with 1-2 messages)
		// and incorrectly wiped the reasoning cache mid-task, breaking the KV-cache
		// prefix for all subsequent turns.
		const convId = this.getConversationId(messages);
		if (convId !== undefined && convId !== this.lastConversationId) {
			if (this.lastConversationId !== undefined) {
				logger.info('New conversation detected — clearing reasoning cache and resetting cache-miss tracker');
			}
			pruneReasoningCache(this.reasoningCache, true);
			this.lastConversationId = convId;
			this.recentCacheMissRates = [];
		}

		// Vision proxy: resolve images → text descriptions before sending to DeepSeek
		const resolvedMessages = await resolveImageMessages(
			messages,
			token,
			() => this.vision.get(),
			progress,
		);
		const deepseekMessages = convertMessages(
			resolvedMessages,
			isThinkingModel,
			this.reasoningCache,
		);
		const tools = modelDef?.capabilities.toolCalling ? convertTools(options.tools) : undefined;

		const totalRequestChars = countRequestChars(deepseekMessages, tools);

		let accumulatedReasoning = '';
		const thinkingPartId = `deepseek-v4-thinking-${Date.now()}-${Math.random()
			.toString(36)
			.slice(2, 9)}`;

		return new Promise<void>((resolve, reject) => {
			client.streamChatCompletion(
				{
					model: getApiModelId(modelInfo.id),
					messages: deepseekMessages,
					stream: true,
					tools,
					tool_choice: getToolChoice(tools, options),
					max_tokens: maxTokens,
					...(isThinkingModel
						? {
								thinking: {
									type: thinkingEffort === 'none' ? ('disabled' as const) : ('enabled' as const),
								},
								...(thinkingEffort === 'none' ? {} : { reasoning_effort: thinkingEffort }),
							}
						: {}),
				},
				{
					onContent: (content: string) => {
						progress.report(new vscode.LanguageModelTextPart(content));
					},

					onThinking: (text: string) => {
						accumulatedReasoning += text;

						// LanguageModelThinkingPart is a proposed API — the class
						// exists at runtime in both stable and Insiders, but the
						// stable vscode.d.ts doesn't include it. The .d.ts
						// augmentation in the project root provides type safety.
						progress.report(
							new vscode.LanguageModelThinkingPart(text, thinkingPartId, {
								provider: PROVIDER_VENDOR,
								model: modelInfo.id,
							}) as unknown as vscode.LanguageModelResponsePart,
						);
					},

					onToolCall: (toolCall: DeepSeekToolCall) => {
						// Cache reasoning keyed by tool_call ID
						if (isThinkingModel && accumulatedReasoning) {
							this.reasoningCache.set(toolCall.id, {
								text: accumulatedReasoning,
								timestamp: Date.now(),
							});
						}

						const args = parseToolInput(toolCall);
						if (!args) {
							reject(
								new Error(
									`DeepSeek V4 returned invalid tool arguments for "${toolCall.function.name}". Retry the request or lower Thinking Effort.`,
								),
							);
							return;
						}

						progress.report(
							new vscode.LanguageModelToolCallPart(toolCall.id, toolCall.function.name, args),
						);
					},

					onError: (error: Error) => {
						reject(error);
					},

					onDone: () => {
						pruneReasoningCache(this.reasoningCache, false);
						resolve();
					},

					onUsage: (usage) => {
						// Accumulate into the session-scoped tracker.
						this.tokenUsageTracker.add(usage);

						this.recordOfficialUsage(usage, totalRequestChars, modelDef.name);
					},
				},
				token,
			);
		});
	}

	async provideTokenCount(
		_modelInfo: vscode.LanguageModelChatInformation,
		text: string | vscode.LanguageModelChatRequestMessage,
		_token: vscode.CancellationToken,
	): Promise<number> {
		if (typeof text === 'string') {
			return Math.max(1, estimateTokens(text));
		}

		if (!text?.content || !Array.isArray(text.content)) {
			return 1;
		}

		// Flatten all text parts into a single string before estimating.
		let combined = '';
		for (const part of text.content) {
			if (part instanceof vscode.LanguageModelTextPart) {
				combined += part.value;
			} else if (part instanceof vscode.LanguageModelToolCallPart) {
				combined += part.name + ' ' + JSON.stringify(part.input) + ' ';
			} else if (part instanceof vscode.LanguageModelToolResultPart) {
				combined += part.callId + ' ';
				for (const item of part.content) {
					combined += estimatePartStr(item) + ' ';
				}
			} else if (part instanceof vscode.LanguageModelDataPart) {
				combined += estimatePartStr(part) + ' ';
			} else if (isThinkingPart(part)) {
				combined += normalizeThinkingValue(part.value) + ' ';
			}
		}
		return Math.max(1, estimateTokens(combined));
	}

	/**
	 * Returns a stable fingerprint of the first user message text (first 256 chars).
	 * Used to detect genuine new conversations rather than relying on message count,
	 * which is unreliable in agent mode where short message sequences are normal mid-task.
	 */
	private getConversationId(
		messages: readonly vscode.LanguageModelChatRequestMessage[],
	): string | undefined {
		for (const msg of messages) {
			if (msg.role !== vscode.LanguageModelChatMessageRole.User) {
				continue;
			}
			for (const part of msg.content) {
				if (part instanceof vscode.LanguageModelTextPart && part.value.trim()) {
					return part.value.slice(0, 256);
				}
			}
		}
		return undefined;
	}

	private async showApiKeyRequiredPrompt(): Promise<void> {
		const action = await vscode.window.showErrorMessage(
			'DeepSeek V4 Bridge needs your DeepSeek API key before Copilot Chat can use it.',
			'Set API Key',
			'Open API Keys',
			'Manage DeepSeek V4 Bridge',
		);

		if (action === 'Set API Key') {
			await this.configureApiKey();
		} else if (action === 'Open API Keys') {
			await vscode.env.openExternal(vscode.Uri.parse('https://platform.deepseek.com/api_keys'));
		} else if (action === 'Manage DeepSeek V4 Bridge') {
			await this.manage();
		}
	}

	private restoreTokenCalibration(): void {
		const saved = this.globalState.get<TokenCalibration>(TOKEN_CALIBRATION_KEY);
		if (!saved || !Number.isFinite(saved.charsPerToken) || saved.charsPerToken <= 0) {
			return;
		}

		this.charsPerToken = saved.charsPerToken;
		this.tokenCalibrationSamples = Math.max(0, saved.samples);
		logger.info(
			`token estimator restored: chars/tok=${this.charsPerToken.toFixed(2)} samples=${this.tokenCalibrationSamples}`,
		);
	}

	private recordOfficialUsage(usage: DeepSeekUsage, requestChars: number, modelName: string): void {
		if (requestChars > 0 && usage.prompt_tokens > 0) {
			const observedRatio = requestChars / usage.prompt_tokens;
			if (observedRatio >= 0.5 && observedRatio <= 12) {
				const alpha = this.tokenCalibrationSamples === 0 ? 1 : 0.25;
				this.charsPerToken = this.charsPerToken * (1 - alpha) + observedRatio * alpha;
				this.tokenCalibrationSamples += 1;

				void this.globalState.update(TOKEN_CALIBRATION_KEY, {
					charsPerToken: this.charsPerToken,
					samples: this.tokenCalibrationSamples,
					updatedAt: Date.now(),
				} satisfies TokenCalibration);
			} else {
				logger.warn(
					`ignored outlier DeepSeek token calibration: chars=${requestChars} prompt_tokens=${usage.prompt_tokens}`,
				);
			}
		}

		const cacheHit = usage.prompt_cache_hit_tokens ?? 0;
		const cacheMiss = usage.prompt_cache_miss_tokens ?? 0;
		const cacheTotal = cacheHit + cacheMiss;
		const hitRate = cacheTotal > 0 ? ((cacheHit / cacheTotal) * 100).toFixed(0) : 'n/a';
		const reasoningTokens = usage.completion_tokens_details?.reasoning_tokens;
		logger.info(
			`Model Turn completed: ${modelName}` +
				` | prompt=${usage.prompt_tokens}` +
				` completion=${usage.completion_tokens}` +
				` total=${usage.total_tokens}` +
				(reasoningTokens === undefined ? '' : ` reasoning=${reasoningTokens}`) +
				` | cache: hit=${cacheHit} miss=${cacheMiss} rate=${hitRate}%` +
				` | context estimator: chars/tok=${this.charsPerToken.toFixed(2)} samples=${this.tokenCalibrationSamples}`,
		);

		// Track rolling cache miss rate to detect KV-cache prefix invalidation.
		// Only count turns with meaningful prompt sizes to avoid false alarms.
		if (cacheTotal > 500) {
			const missRate = cacheMiss / cacheTotal;
			this.recentCacheMissRates.push(missRate);
			if (this.recentCacheMissRates.length > 5) {
				this.recentCacheMissRates.shift();
			}

			if (this.recentCacheMissRates.length >= 3) {
				const avgMissRate =
					this.recentCacheMissRates.reduce((a, b) => a + b, 0) /
					this.recentCacheMissRates.length;
				if (avgMissRate >= 0.8) {
					logger.warn(
						`Cache miss anomaly: avg miss rate ${(avgMissRate * 100).toFixed(0)}% over ${this.recentCacheMissRates.length} turns` +
							` — prompt prefix may be unstable. This inflates costs significantly.` +
							` Causes: reasoning cache wiped mid-session, tool arg serialization changed, or very long conversation. Consider starting a new conversation.`,
					);
				}
			}
		}
	}
}

// ---- Helpers ----

function toChatInfo(m: ModelDefinition, hasApiKey: boolean): ModelPickerChatInformation {
	return {
		id: m.id,
		name: m.name,
		family: m.family,
		version: m.version,
		detail: hasApiKey ? m.detail : API_KEY_REQUIRED_DETAIL,
		tooltip: hasApiKey ? undefined : API_KEY_REQUIRED_DETAIL,
		statusIcon: hasApiKey ? undefined : new vscode.ThemeIcon('warning'),
		maxInputTokens: m.maxInputTokens,
		maxOutputTokens: m.maxOutputTokens,
		isUserSelectable: true,
		capabilities: {
			toolCalling: m.capabilities.toolCalling,
			imageInput: m.capabilities.imageInput,
		},
		...(m.capabilities.thinking
			? { configurationSchema: THINKING_EFFORT_CONFIGURATION_SCHEMA }
			: {}),
	};
}

function getConfiguredThinkingEffort(options: ModelConfigurationOptions): ThinkingEffort {
	const configuredEffort =
		options.modelConfiguration?.reasoningEffort ?? options.configuration?.reasoningEffort;

	if (configuredEffort === 'none') {
		return 'none';
	}

	if (configuredEffort === 'high') {
		return 'high';
	}

	return configuredEffort === 'max' ? 'max' : 'high';
}

function getToolChoice(
	tools: DeepSeekTool[] | undefined,
	options: vscode.ProvideLanguageModelChatResponseOptions,
): 'auto' | 'required' | undefined {
	if (!tools || tools.length === 0) {
		return undefined;
	}

	return options.toolMode === vscode.LanguageModelChatToolMode.Required ? 'required' : 'auto';
}

function parseToolInput(toolCall: DeepSeekToolCall): object | undefined {
	const raw = toolCall.function.arguments.trim();
	if (!raw) {
		return {};
	}

	try {
		const parsed = JSON.parse(raw);
		if (parsed && typeof parsed === 'object' && !Array.isArray(parsed)) {
			return parsed as Record<string, unknown>;
		}
		logger.warn(
			`DeepSeek V4 returned non-object tool arguments for ${toolCall.function.name}: ${raw.slice(0, 200)}`,
		);
		return undefined;
	} catch (error) {
		logger.warn(
			`DeepSeek V4 returned malformed tool arguments for ${toolCall.function.name}: ${raw.slice(0, 200)}`,
			error,
		);
		return undefined;
	}
}

function estimatePartStr(part: unknown): string {
	if (part instanceof vscode.LanguageModelTextPart) {
		return part.value;
	}

	if (part instanceof vscode.LanguageModelDataPart) {
		if (part.mimeType.startsWith('text/') || part.mimeType.includes('json')) {
			return new TextDecoder().decode(part.data);
		}
		return '';
	}

	if (isThinkingPart(part)) {
		return normalizeThinkingValue(part.value);
	}

	if (typeof part === 'string') {
		return part;
	}

	if (part === undefined || part === null) {
		return '';
	}

	try {
		return JSON.stringify(part);
	} catch {
		return String(part);
	}
}

/**
 * Estimate tokens for a string using character-class heuristics.
 *
 * DeepSeek V4 uses a cl100k_base-compatible tokenizer.
 * Heuristic breakdown by character class:
 * - Alphanumeric/underscore (code identifiers): ~3.5 chars per token
 * - Whitespace/newlines: ~0.3 tokens each
 * - Punctuation/operators: ~1 token each
 * - Non-ASCII (CJK, emoji, etc): ~1.5 chars per token
 */
function estimateTokens(str: string): number {
	let tokens = 0;
	let i = 0;
	while (i < str.length) {
		const code = str.charCodeAt(i);
		if (
			(code >= 65 && code <= 90) || // A-Z
			(code >= 97 && code <= 122) || // a-z
			(code >= 48 && code <= 57) || // 0-9
			code === 95 // _
		) {
			// Run of identifier chars — batch them
			let run = 1;
			while (i + run < str.length) {
				const c2 = str.charCodeAt(i + run);
				if (
					(c2 >= 65 && c2 <= 90) ||
					(c2 >= 97 && c2 <= 122) ||
					(c2 >= 48 && c2 <= 57) ||
					c2 === 95
				) {
					run++;
				} else {
					break;
				}
			}
			tokens += Math.ceil(run / 3.5);
			i += run;
		} else if (code === 32 || code === 9) {
			// Space or tab
			tokens += 0.3;
			i++;
		} else if (code === 10 || code === 13) {
			// Newline or carriage return
			tokens += 0.4;
			i++;
		} else if (code < 128) {
			// ASCII punctuation / operators
			tokens += 1;
			i++;
		} else {
			// Non-ASCII: consume 1-2 chars, count as ~0.7 tokens
			tokens += 0.7;
			i++;
		}
	}
	return Math.ceil(tokens);
}

function isThinkingPart(part: unknown): part is { value: string | string[] } {
	const ctor = (
		vscode as unknown as {
			LanguageModelThinkingPart?: new (...args: never[]) => unknown;
		}
	).LanguageModelThinkingPart;
	if (ctor && part instanceof ctor) {
		return true;
	}

	return (
		typeof part === 'object' &&
		part !== null &&
		(part as { constructor?: { name?: string } }).constructor?.name ===
			'LanguageModelThinkingPart' &&
		'value' in part
	);
}

function normalizeThinkingValue(value: string | string[]): string {
	return Array.isArray(value) ? value.join('') : value;
}
