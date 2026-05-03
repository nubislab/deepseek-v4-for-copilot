import type { ModelDefinition } from './types';

/**
 * Compile-time constants shared across the extension.
 *
 * These do NOT depend on the VS Code runtime (no workspace configuration,
 * no secrets API). For run-time settings reads see `config.ts`.
 */

/** VS Code configuration section prefix for all extension settings. */
export const CONFIG_SECTION = 'deepseek-v4-bridge';

// ---- Secret keys ----

/** SecretStorage key for the DeepSeek API key. */
export const API_KEY_SECRET = 'deepseek-v4-bridge.apiKey';

/** memento key tracking whether the welcome walkthrough has been shown. */
export const WELCOME_SHOWN_KEY = 'deepseek-v4-bridge.welcomeShown';

/** memento key storing token-count calibration learned from official DeepSeek usage. */
export const TOKEN_CALIBRATION_KEY = 'deepseek-v4-bridge.tokenCalibration';

// ---- Walkthrough ----

/** Walkthrough contribution ID. */
export const WALKTHROUGH_ID = 'Vizards.deepseek-v4-bridge-local#deepseekV4BridgeGettingStarted';

/** Distinct command prefix used by this extension. */
export const COMMAND_PREFIX = 'deepseek-v4-bridge';

/** Language model vendor ID registered by this extension. */
export const PROVIDER_VENDOR = 'deepseek-v4-bridge';

// ---- Model picker ----

/** Detail text shown in the model picker when no API key is configured. */
export const API_KEY_REQUIRED_DETAIL =
	'Please run DeepSeek V4 Bridge: Set API Key to configure.';

/** Per-model configuration schema consumed by Copilot Chat's model picker. */
export const THINKING_EFFORT_CONFIGURATION_SCHEMA = {
	properties: {
		reasoningEffort: {
			type: 'string',
			title: 'Thinking Effort',
			enum: ['none', 'high', 'max'],
			enumItemLabels: ['None', 'High', 'Max'],
			enumDescriptions: [
				'Disable thinking for faster responses',
				'Recommended for most tasks',
				'Maximum reasoning depth for complex agent tasks',
			],
			default: 'high',
			group: 'navigation',
		},
	},
} as const;

/** DeepSeek V4 output limit. Thinking tokens are included in this budget. */
export const DEEPSEEK_V4_MAX_OUTPUT_TOKENS = 65536;

// ---- Vision proxy ----

/** Preferred model ID used first for the vision proxy when it is available. */
export const DEFAULT_VISION_MODEL_ID = 'oswe-vscode-prime';

/**
 * Prompt sent to the vision proxy model when describing image attachments
 * before forwarding them to text-only DeepSeek models.
 */
export const IMAGE_DESCRIPTION_PROMPT =
	'Describe the visual contents of this image in detail, including any text, objects, people, or context that would be relevant for understanding it. Focus on factual visual elements.';

// ---- Cache ----

/** Max entries in the reasoning-content cache before eviction kicks in. */
export const MAX_CACHE_SIZE = 200;

// ---- Model registry ----

/** Available DeepSeek models exposed through the language model provider. */
export const MODELS: ModelDefinition[] = [
	{
		id: 'deepseek-v4-flash',
		name: 'DeepSeek V4 Flash',
		family: 'deepseek',
		version: 'v4',
		detail: 'Fast V4 coding model - 1M context - thinking/tools',
		maxInputTokens: 1048576,
		maxOutputTokens: DEEPSEEK_V4_MAX_OUTPUT_TOKENS,
		capabilities: {
			toolCalling: true,
			imageInput: true,
			thinking: true,
		},
		requiresThinkingParam: true,
	},
	{
		id: 'deepseek-v4-pro',
		name: 'DeepSeek V4 Pro',
		family: 'deepseek',
		version: 'v4',
		detail: 'Most capable V4 reasoning model - 1M context - thinking/tools',
		maxInputTokens: 1048576,
		maxOutputTokens: DEEPSEEK_V4_MAX_OUTPUT_TOKENS,
		capabilities: {
			toolCalling: true,
			imageInput: true,
			thinking: true,
		},
		requiresThinkingParam: true,
	},
];
