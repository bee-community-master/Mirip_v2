// @vitest-environment jsdom

import { describe, expect, it, vi } from 'vitest';
import { focusInitialElement, getFocusableElements, trapFocus } from '$lib/utils/focus';

describe('focus utilities', () => {
	it('returns focusable elements in DOM order', () => {
		const container = document.createElement('div');
		container.innerHTML = `
			<button type="button">First</button>
			<div tabindex="-1">Ignored</div>
			<a href="/next">Second</a>
		`;

		expect(getFocusableElements(container).map((element) => element.textContent?.trim())).toEqual([
			'First',
			'Second'
		]);
	});

	it('focuses the first focusable element when available', () => {
		const container = document.createElement('div');
		container.innerHTML = `
			<div tabindex="-1">Ignored</div>
			<button type="button">First</button>
			<button type="button">Second</button>
		`;
		document.body.append(container);

		focusInitialElement(container);

		expect(document.activeElement).toBe(container.querySelector('button'));
		container.remove();
	});

	it('cycles focus back to the first element when tabbing from the last', () => {
		const container = document.createElement('div');
		container.innerHTML = `
			<button type="button">First</button>
			<button type="button">Second</button>
		`;
		document.body.append(container);

		const buttons = container.querySelectorAll('button');
		buttons[1].focus();

		const event = new KeyboardEvent('keydown', { key: 'Tab' });
		Object.defineProperty(event, 'preventDefault', {
			value: vi.fn()
		});

		trapFocus(container, event);

		expect(document.activeElement).toBe(buttons[0]);
		container.remove();
	});
});
