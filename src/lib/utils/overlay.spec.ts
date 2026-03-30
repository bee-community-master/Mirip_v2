// @vitest-environment jsdom

import { describe, expect, it, vi } from 'vitest';
import {
	focusOverlayContainer,
	getActiveElement,
	handleOverlayKeydown,
	lockBodyScroll,
	restoreFocus
} from '$lib/utils/overlay';

describe('overlay utilities', () => {
	it('returns the current active element when one exists', () => {
		const button = document.createElement('button');
		document.body.append(button);
		button.focus();

		expect(getActiveElement()).toBe(button);

		button.remove();
	});

	it('locks body scroll and restores the previous overflow value', () => {
		document.body.style.overflow = 'auto';

		const restoreScroll = lockBodyScroll();

		expect(document.body.style.overflow).toBe('hidden');

		restoreScroll();

		expect(document.body.style.overflow).toBe('auto');
	});

	it('focuses the first focusable element in the overlay container', async () => {
		const container = document.createElement('div');
		container.innerHTML = `
			<div tabindex="-1">Ignored</div>
			<button type="button">Open</button>
			<button type="button">Close</button>
		`;
		document.body.append(container);

		await focusOverlayContainer(container);

		expect(document.activeElement).toBe(container.querySelector('button'));
		container.remove();
	});

	it('restores focus to the previous element when provided', () => {
		const trigger = document.createElement('button');
		document.body.append(trigger);

		restoreFocus(trigger);

		expect(document.activeElement).toBe(trigger);
		trigger.remove();
	});

	it('closes the overlay on Escape and traps focus on Tab', () => {
		const onClose = vi.fn();
		const container = document.createElement('div');
		container.innerHTML = `
			<button type="button">First</button>
			<button type="button">Second</button>
		`;
		document.body.append(container);

		const buttons = container.querySelectorAll('button');
		buttons[1].focus();

		const escapeEvent = new KeyboardEvent('keydown', { key: 'Escape' });
		Object.defineProperty(escapeEvent, 'preventDefault', {
			value: vi.fn()
		});
		handleOverlayKeydown({ open: true, container, onClose }, escapeEvent);

		expect(onClose).toHaveBeenCalledTimes(1);

		const tabEvent = new KeyboardEvent('keydown', { key: 'Tab' });
		Object.defineProperty(tabEvent, 'preventDefault', {
			value: vi.fn()
		});
		handleOverlayKeydown({ open: true, container, onClose }, tabEvent);

		expect(document.activeElement).toBe(buttons[0]);
		container.remove();
	});
});
