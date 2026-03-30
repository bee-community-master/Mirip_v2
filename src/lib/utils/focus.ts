const FOCUSABLE_SELECTOR = [
	'a[href]',
	'button:not([disabled])',
	'input:not([disabled]):not([type="hidden"])',
	'select:not([disabled])',
	'textarea:not([disabled])',
	'[tabindex]:not([tabindex="-1"])'
].join(',');

export function getFocusableElements(container: HTMLElement) {
	return Array.from(container.querySelectorAll<HTMLElement>(FOCUSABLE_SELECTOR)).filter(
		(element) => element.getAttribute('aria-hidden') !== 'true' && element.tabIndex >= 0
	);
}

export function focusInitialElement(container: HTMLElement) {
	const [firstElement] = getFocusableElements(container);
	(firstElement ?? container).focus();
}

export function trapFocus(container: HTMLElement, event: KeyboardEvent) {
	if (event.key !== 'Tab') {
		return;
	}

	const focusableElements = getFocusableElements(container);

	if (!focusableElements.length) {
		container.focus();
		event.preventDefault();
		return;
	}

	const firstElement = focusableElements[0];
	const lastElement = focusableElements[focusableElements.length - 1];
	const activeElement =
		document.activeElement instanceof HTMLElement ? document.activeElement : null;

	if (!activeElement || !container.contains(activeElement)) {
		(event.shiftKey ? lastElement : firstElement).focus();
		event.preventDefault();
		return;
	}

	if (event.shiftKey && (activeElement === firstElement || activeElement === container)) {
		lastElement.focus();
		event.preventDefault();
		return;
	}

	if (!event.shiftKey && activeElement === lastElement) {
		firstElement.focus();
		event.preventDefault();
	}
}
