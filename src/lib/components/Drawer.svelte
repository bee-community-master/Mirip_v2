<script lang="ts">
	import type { Snippet } from 'svelte';
	import { fade, fly } from 'svelte/transition';
	import { X } from 'lucide-svelte';
	import {
		focusOverlayContainer,
		getActiveElement,
		handleOverlayKeydown,
		lockBodyScroll,
		restoreFocus
	} from '$lib/utils/overlay';

	let {
		open,
		title = '',
		onClose,
		children
	}: {
		open: boolean;
		title?: string;
		onClose: () => void;
		children?: Snippet;
	} = $props();

	let panelElement = $state<HTMLDivElement | null>(null);
	let previousFocusedElement = $state<HTMLElement | null>(null);

	$effect(() => {
		if (!open) {
			return;
		}

		previousFocusedElement = getActiveElement();
		const restoreScroll = lockBodyScroll();

		void focusOverlayContainer(panelElement);

		return () => {
			restoreScroll();
			restoreFocus(previousFocusedElement);
		};
	});

	function handleKeydown(event: KeyboardEvent) {
		handleOverlayKeydown({ open, container: panelElement, onClose }, event);
	}
</script>

<svelte:window onkeydown={handleKeydown} />

{#if open}
	<div
		class="fixed inset-0 z-[85] flex justify-end bg-night-950/70 backdrop-blur-sm"
		transition:fade={{ duration: 180 }}
	>
		<button
			type="button"
			tabindex="-1"
			class="absolute inset-0 cursor-default"
			onclick={onClose}
			aria-label="패널 닫기"
		></button>

		<div
			bind:this={panelElement}
			role="dialog"
			aria-modal="true"
			aria-label={title || 'Drawer'}
			tabindex="-1"
			class="glass-panel no-scrollbar relative h-full w-full overflow-y-auto overscroll-contain border-l border-white/10 px-5 py-6 shadow-[0_36px_120px_rgba(0,0,0,0.55)] sm:max-w-xl sm:px-7"
			transition:fly={{ duration: 240, x: 36 }}
		>
			<div class="mb-8 flex items-start justify-between gap-4">
				<div>
					<p class="mb-2 text-xs font-bold uppercase tracking-[0.3em] text-white/35">Competition Detail</p>
					<h3 class="font-display text-3xl font-black tracking-[-0.05em] text-white">{title}</h3>
				</div>
				<button
					type="button"
					class="rounded-full border border-white/10 bg-white/5 p-2 text-white/70 transition hover:bg-white/10 hover:text-white focus-visible:outline focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-fuchsia-300"
					onclick={onClose}
					aria-label="닫기"
				>
					<X class="size-4" aria-hidden="true" />
				</button>
			</div>

			{@render children?.()}
		</div>
	</div>
{/if}
