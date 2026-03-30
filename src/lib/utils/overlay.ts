import { tick } from 'svelte';
import { focusInitialElement, trapFocus } from '$lib/utils/focus';

export function getActiveElement() {
	return document.activeElement instanceof HTMLElement ? document.activeElement : null;
}

export function lockBodyScroll() {
	const previousOverflow = document.body.style.overflow;
	document.body.style.overflow = 'hidden';

	return () => {
		document.body.style.overflow = previousOverflow;
	};
}

export async function focusOverlayContainer(container: HTMLElement | null) {
	await tick();

	if (container) {
		focusInitialElement(container);
	}
}

export function restoreFocus(element: HTMLElement | null) {
	element?.focus();
}

export function handleOverlayKeydown(
	{
		open,
		container,
		onClose
	}: {
		open: boolean;
		container: HTMLElement | null;
		onClose: () => void;
	},
	event: KeyboardEvent
) {
	if (!open) {
		return;
	}

	if (event.key === 'Escape') {
		event.preventDefault();
		onClose();
		return;
	}

	if (container) {
		trapFocus(container, event);
	}
}
