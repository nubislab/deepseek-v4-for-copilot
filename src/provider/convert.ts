import vscode from 'vscode';
import { logger } from '../logger';
import type { DeepSeekMessage, DeepSeekTool, DeepSeekToolCall } from '../types';
import type { ReasoningEntry } from './cache';

/**
 * Stable JSON serialization with sorted object keys.
 * Ensures tool call argument strings are byte-identical for the same logical input
 * across turns, preventing DeepSeek KV cache invalidation from key-ordering variance.
 *
 * Matches JSON.stringify behaviour:
 * - Object keys with `undefined` values are skipped.
 * - Array elements that are `undefined` are serialized as `null`.
 */
function stableStringify(val: unknown): string {
	if (val === undefined) {
		// Reached only from array elements; JSON.stringify coerces these to null.
		return 'null';
	}
	if (val === null || typeof val !== 'object') {
		// JSON.stringify cannot return undefined here because undefined is handled above.
		return JSON.stringify(val) as string;
	}
	if (Array.isArray(val)) {
		return '[' + val.map(stableStringify).join(',') + ']';
	}
	const keys = Object.keys(val as object).sort();
	const pairs: string[] = [];
	for (const k of keys) {
		const v = (val as Record<string, unknown>)[k];
		if (v === undefined) {
			continue; // skip undefined values, matching JSON.stringify object behaviour
		}
		pairs.push(JSON.stringify(k) + ':' + stableStringify(v));
	}
	return '{' + pairs.join(',') + '}';
}

/**
 * Convert VS Code chat messages to DeepSeek format.
 * Injects cached reasoning_content for assistant messages that had tool calls
 * in prior turns.
 */
export function convertMessages(
	messages: readonly vscode.LanguageModelChatRequestMessage[],
	isThinkingModel: boolean,
	reasoningCache: Map<string, ReasoningEntry>,
): DeepSeekMessage[] {
	const result: DeepSeekMessage[] = [];

	for (const message of messages) {
		const role = mapRole(message.role);
		if (!role) {
			logger.warn(`Skipping unsupported VS Code chat message role: ${String(message.role)}`);
			continue;
		}

		let content = '';
		let reasoningFromParts = '';
		const toolCalls: DeepSeekToolCall[] = [];
		const toolResults: Array<{ callId: string; content: string }> = [];

		for (const part of message.content) {
			if (part instanceof vscode.LanguageModelTextPart) {
				content += part.value;
			} else if (isThinkingPart(part)) {
				reasoningFromParts += normalizeThinkingValue(part.value);
			} else if (part instanceof vscode.LanguageModelToolCallPart) {
				toolCalls.push({
					id: part.callId,
					type: 'function',
					function: {
						name: part.name,
						// stableStringify ensures key ordering is deterministic across turns,
						// so the serialized argument string never changes for the same logical
						// input — preventing DeepSeek KV cache prefix invalidation.
						arguments: stableStringify(part.input),
					},
				});
			} else if (part instanceof vscode.LanguageModelToolResultPart) {
				let toolContent = '';
				for (const item of part.content) {
					if (item instanceof vscode.LanguageModelTextPart) {
						toolContent += item.value;
					} else {
						toolContent += serializeToolResultItem(item);
					}
				}
				toolResults.push({
					callId: part.callId,
					content: toolContent || '[Tool result: empty]',
				});
			}
		}

		if (role === 'assistant') {
			// Inject reasoning_content from cache for assistant messages
			// that have tool calls (per DeepSeek API requirement), and reuse
			// persisted ThinkingPart content when Copilot gives it back.
			let reasoningContent = reasoningFromParts || undefined;
			if (isThinkingModel && toolCalls.length > 0 && !reasoningContent) {
				for (const tc of toolCalls) {
					const cached = reasoningCache.get(tc.id);
					if (cached) {
						reasoningContent = cached.text;
						break;
					}
				}
			}

			if (content || toolCalls.length > 0) {
				const msg: DeepSeekMessage = {
					role: 'assistant' as const,
					content: content || '',
				};

				if (toolCalls.length > 0) {
					msg.tool_calls = toolCalls;
				}

				if (isThinkingModel && (reasoningContent || toolCalls.length > 0)) {
					msg.reasoning_content = reasoningContent || '';
				}

				result.push(msg);
			}
		} else if (content) {
			result.push({
				role,
				content: content,
			});
		}

		// Tool result messages follow their associated assistant message
		for (const tr of toolResults) {
			result.push({
				role: 'tool',
				content: tr.content,
				tool_call_id: tr.callId,
			});
		}
	}

	return result;
}

function mapRole(
	role: vscode.LanguageModelChatMessageRole,
): 'user' | 'assistant' | 'system' | undefined {
	switch (role) {
		case vscode.LanguageModelChatMessageRole.User:
			return 'user';
		case vscode.LanguageModelChatMessageRole.Assistant:
			return 'assistant';
		default:
			break;
	}

	const roles = vscode.LanguageModelChatMessageRole as unknown as Record<string, unknown>;
	if (role === roles.System || String(role).toLowerCase() === 'system') {
		return 'system';
	}

	return undefined;
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

function serializeToolResultItem(item: unknown): string {
	if (item instanceof vscode.LanguageModelDataPart) {
		if (item.mimeType.startsWith('text/') || item.mimeType.includes('json')) {
			return new TextDecoder().decode(item.data);
		}
		return `[${item.mimeType} data omitted]`;
	}

	if (typeof item === 'string') {
		return item;
	}

	if (item === undefined || item === null) {
		return '';
	}

	try {
		return JSON.stringify(item);
	} catch {
		return String(item);
	}
}

/**
 * Convert VS Code tool definitions to DeepSeek format.
 */
export function convertTools(
	tools: readonly vscode.LanguageModelChatTool[] | undefined,
): DeepSeekTool[] | undefined {
	if (!tools || tools.length === 0) {
		return undefined;
	}

	return tools.map((tool) => ({
		type: 'function' as const,
		function: {
			name: tool.name,
			description: tool.description,
			parameters: tool.inputSchema as Record<string, unknown> | undefined,
		},
	}));
}

/**
 * Count total characters across all messages to calibrate chars-per-token ratio.
 */
export function countMessageChars(messages: DeepSeekMessage[]): number {
	let total = 0;
	for (const msg of messages) {
		total += msg.content?.length ?? 0;
		if (msg.tool_calls) {
			for (const tc of msg.tool_calls) {
				total += tc.function?.name?.length ?? 0;
				total += tc.function?.arguments?.length ?? 0;
			}
		}
	}
	return total;
}

/**
 * Count request characters used to calibrate token estimates against official
 * DeepSeek usage. Tool schemas are included because DeepSeek bills/counts them
 * as prompt tokens when tool calling is enabled.
 */
export function countRequestChars(
	messages: DeepSeekMessage[],
	tools: DeepSeekTool[] | undefined,
): number {
	let total = countMessageChars(messages);
	if (tools && tools.length > 0) {
		total += JSON.stringify(tools).length;
	}
	return total;
}
