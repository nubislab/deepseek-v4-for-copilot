import vscode from 'vscode';
import {
	CONFIG_SECTION,
	DEFAULT_VISION_MODEL_ID,
	IMAGE_DESCRIPTION_PROMPT,
	PROVIDER_VENDOR,
} from '../consts';
import { logger } from '../logger';

/**
 * Resolve any image parts in user messages by forwarding them to a vision
 * model and replacing them with text descriptions. This lets text-only models
 * like DeepSeek effectively "see" images.
 */
export async function resolveImageMessages(
	messages: readonly vscode.LanguageModelChatRequestMessage[],
	token: vscode.CancellationToken,
	getModels: () => Promise<readonly vscode.LanguageModelChat[]>,
	progress?: vscode.Progress<vscode.LanguageModelResponsePart>,
): Promise<readonly vscode.LanguageModelChatRequestMessage[]> {
	const hasImages = messages.some((m) => m.content.some((p) => isImageDataPart(p)));
	if (!hasImages) {
		return messages;
	}

	const visionModels = await getModels();
	if (visionModels.length === 0) {
		logger.warn('No vision proxy model available; replacing images with a text notice.');
	}

	const result: vscode.LanguageModelChatRequestMessage[] = [];

	for (const message of messages) {
		const imageParts: vscode.LanguageModelDataPart[] = [];
		const otherParts: vscode.LanguageModelInputPart[] = [];

		for (const part of message.content as readonly vscode.LanguageModelInputPart[]) {
			if (isImageDataPart(part)) {
				imageParts.push(part);
			} else {
				otherParts.push(part);
			}
		}

		if (imageParts.length === 0) {
			result.push(message as vscode.LanguageModelChatRequestMessage);
			continue;
		}

		reportVisionProgress(progress, imageParts.length);
		const description = await describeImages(imageParts, otherParts, visionModels, token);
		if (description) {
			otherParts.push(
				new vscode.LanguageModelTextPart(
					`[Image description for DeepSeek V4: ${description.trim()}]`,
				),
			);
		} else {
			otherParts.push(
				new vscode.LanguageModelTextPart(
					'[Image: DeepSeek V4 is text-only and no working vision proxy model could describe this attachment. Ask the user to run "DeepSeek V4 Bridge: Set Vision Proxy Model".]',
				),
			);
		}

		result.push({
			role: message.role,
			content: otherParts,
		} as unknown as vscode.LanguageModelChatRequestMessage);
	}

	return result;
}

function reportVisionProgress(
	progress: vscode.Progress<vscode.LanguageModelResponsePart> | undefined,
	imageCount: number,
): void {
	if (!progress) {
		return;
	}

	progress.report(
		new vscode.LanguageModelThinkingPart(
			`Describing ${imageCount === 1 ? 'image attachment' : `${imageCount} image attachments`} for DeepSeek V4...`,
			`deepseek-v4-vision-${Date.now()}-${Math.random().toString(36).slice(2, 9)}`,
			{
				provider: PROVIDER_VENDOR,
				phase: 'vision-proxy',
			},
		) as unknown as vscode.LanguageModelResponsePart,
	);
}

/**
 * Get the vision proxy model. Cached after first lookup.
 * Uses the configured model ID, or defaults to DEFAULT_VISION_MODEL_ID.
 */
export function createVisionModelGetter(): {
	get: () => Promise<readonly vscode.LanguageModelChat[]>;
	reset: () => void;
} {
	let visionModels: readonly vscode.LanguageModelChat[] | undefined;
	let visionModelPromise: Promise<readonly vscode.LanguageModelChat[]> | undefined;

	return {
		async get() {
			if (visionModels) {
				return visionModels;
			}
			if (visionModelPromise) {
				return visionModelPromise;
			}

			visionModelPromise = (async () => {
				const candidates = await getVisionModelCandidates();
				if (candidates.length > 0) {
					logger.info(`Vision proxy candidates: ${candidates.map((m) => m.id).join(', ')}`);
				} else {
					logger.warn('No language models found for vision proxy.');
				}
				visionModels = candidates;
				return candidates;
			})();

			return visionModelPromise;
		},

		reset() {
			visionModels = undefined;
			visionModelPromise = undefined;
		},
	};
}

/**
 * Let the user pick which model to use for describing image attachments.
 */
export async function setVisionProxyModel(): Promise<void> {
	const allModels = await vscode.lm.selectChatModels();
	const candidates = allModels.filter((m) => m.vendor !== 'deepseek' && m.vendor !== PROVIDER_VENDOR);

	if (candidates.length === 0) {
		vscode.window.showInformationMessage(
			'No language models available in your VS Code environment.',
		);
		return;
	}

	const currentId = getConfiguredVisionModelId();

	const items = [
		{
			label: '$(sparkle) Auto-detect',
			description: currentId ? undefined : 'current',
			detail: `Try ${DEFAULT_VISION_MODEL_ID} first, then fall back to another installed Copilot vision-capable model.`,
			id: '',
		},
		...candidates.map((m) => ({
			label: m.id,
			description: `vendor: ${m.vendor}`,
			detail: m.id === currentId ? '✓ current' : undefined,
			id: m.id,
		})),
	];

	const picked = await vscode.window.showQuickPick(items, {
		placeHolder: 'Pick a model to describe image attachments before DeepSeek V4 sees them',
		matchOnDescription: true,
	});

	if (picked) {
		const config = vscode.workspace.getConfiguration(CONFIG_SECTION);
		await config.update('visionModel', picked.id, vscode.ConfigurationTarget.Global);
	}
}

export function getConfiguredVisionModelId(): string | undefined {
	const config = vscode.workspace.getConfiguration(CONFIG_SECTION);
	const id = config.get<string>('visionModel', '');
	return id.trim() || undefined;
}

function isImageDataPart(part: unknown): part is vscode.LanguageModelDataPart {
	return part instanceof vscode.LanguageModelDataPart && part.mimeType.startsWith('image/');
}

function getVisionPrompt(): string {
	const config = vscode.workspace.getConfiguration(CONFIG_SECTION);
	return (
		config.get<string>('visionPrompt', IMAGE_DESCRIPTION_PROMPT).trim() || IMAGE_DESCRIPTION_PROMPT
	);
}

async function describeImages(
	imageParts: readonly vscode.LanguageModelDataPart[],
	otherParts: readonly vscode.LanguageModelInputPart[],
	models: readonly vscode.LanguageModelChat[],
	token: vscode.CancellationToken,
): Promise<string | undefined> {
	if (models.length === 0) {
		return undefined;
	}

	const context = otherParts
		.filter((p): p is vscode.LanguageModelTextPart => p instanceof vscode.LanguageModelTextPart)
		.map((p) => p.value.trim())
		.filter(Boolean)
		.join('\n\n');
	const prompt = context
		? `${getVisionPrompt()}\n\nUser message context:\n${context}`
		: getVisionPrompt();

	for (const model of models) {
		try {
			const visionMsg = vscode.LanguageModelChatMessage.User([
				...imageParts,
				new vscode.LanguageModelTextPart(prompt),
			] as (vscode.LanguageModelDataPart | vscode.LanguageModelTextPart)[]);

			const response = await model.sendRequest(
				[visionMsg],
				{
					justification:
						'Describe image attachments so DeepSeek V4 can answer inside Copilot Chat.',
				},
				token,
			);
			let description = '';
			for await (const chunk of response.stream) {
				if (chunk instanceof vscode.LanguageModelTextPart) {
					description += chunk.value;
				}
			}

			const trimmed = description.trim();
			if (trimmed) {
				logger.info(`Vision proxy succeeded with ${model.id}`);
				return trimmed;
			}
		} catch (err) {
			logger.warn(`Vision proxy model ${model.id} failed`, err);
		}
	}

	return undefined;
}

async function getVisionModelCandidates(): Promise<readonly vscode.LanguageModelChat[]> {
	const configuredId = getConfiguredVisionModelId();
	const candidates: vscode.LanguageModelChat[] = [];

	if (configuredId) {
		appendUniqueModels(candidates, await selectNonDeepSeekModels({ id: configuredId }));
		if (candidates.length === 0) {
			logger.warn(`Configured vision model "${configuredId}" was not found.`);
		}
	}

	if (DEFAULT_VISION_MODEL_ID !== configuredId) {
		appendUniqueModels(candidates, await selectNonDeepSeekModels({ id: DEFAULT_VISION_MODEL_ID }));
	}

	const allModels = await vscode.lm.selectChatModels();
	appendUniqueModels(
		candidates,
		allModels
			.filter((m) => m.vendor !== 'deepseek')
			.sort((a, b) => getVisionPreferenceScore(a) - getVisionPreferenceScore(b)),
	);

	return candidates;
}

async function selectNonDeepSeekModels(
	selector: vscode.LanguageModelChatSelector,
): Promise<vscode.LanguageModelChat[]> {
	const models = await vscode.lm.selectChatModels(selector);
	return models.filter((m) => m.vendor !== 'deepseek');
}

function appendUniqueModels(
	target: vscode.LanguageModelChat[],
	models: readonly vscode.LanguageModelChat[],
): void {
	for (const model of models) {
		if (!target.some((existing) => existing.id === model.id && existing.vendor === model.vendor)) {
			target.push(model);
		}
	}
}

function getVisionPreferenceScore(model: vscode.LanguageModelChat): number {
	const text = `${model.id} ${model.name} ${model.family}`.toLowerCase();
	if (text.includes('vision') || text.includes('gpt-4o')) return 0;
	if (text.includes('gpt-5')) return 1;
	if (text.includes('claude') || text.includes('sonnet')) return 2;
	if (text.includes('gemini')) return 3;
	return 10;
}
