<script lang="ts">
	import type { Snippet } from 'svelte';
	import { fade, scale } from 'svelte/transition';
	import { X } from 'lucide-svelte';

	let {
		open,
		title = '',
		onClose,
		className = '',
		children
	}: {
		open: boolean;
		title?: string;
		onClose: () => void;
		className?: string;
		children?: Snippet;
	} = $props();

	function handleKeydown(event: KeyboardEvent) {
		if (open && event.key === 'Escape') {
			onClose();
		}
	}
</script>

<svelte:window onkeydown={handleKeydown} />

{#if open}
	<div
		class="fixed inset-0 z-[80] flex items-center justify-center bg-night-950/72 px-4 py-6 backdrop-blur-md"
		transition:fade={{ duration: 180 }}
	>
		<button
			type="button"
			class="absolute inset-0 cursor-default"
			onclick={onClose}
			aria-label="모달 닫기"
		></button>

		<div
			role="dialog"
			aria-modal="true"
			class={`glass-panel relative w-full max-w-2xl overscroll-contain rounded-[32px] p-6 shadow-[0_36px_120px_rgba(0,0,0,0.55)] sm:p-8 ${className}`}
			transition:scale={{ duration: 220, start: 0.94 }}
		>
			<div class="mb-6 flex items-start justify-between gap-4">
				{#if title}
					<h3 class="font-display text-2xl font-bold tracking-[-0.04em] text-white">{title}</h3>
				{/if}
				<button
					type="button"
					class="rounded-full border border-white/10 bg-white/5 p-2 text-white/70 transition hover:bg-white/10 hover:text-white"
					onclick={onClose}
					aria-label="닫기"
				>
					<X class="size-4" />
				</button>
			</div>

			{@render children?.()}
		</div>
	</div>
{/if}
