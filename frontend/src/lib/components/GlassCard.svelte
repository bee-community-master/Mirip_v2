<script lang="ts">
	import type { Snippet } from 'svelte';

	let {
		children,
		className = '',
		hoverable = false
	}: {
		children?: Snippet;
		className?: string;
		hoverable?: boolean;
	} = $props();
</script>

<div
	class={`glass-panel surface-outline rounded-[28px] ${hoverable ? 'transition duration-300 hover:-translate-y-1 hover:bg-white/10' : ''} ${className}`}
>
	{@render children?.()}
</div>
