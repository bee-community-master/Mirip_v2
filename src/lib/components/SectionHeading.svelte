<script lang="ts">
	let {
		badge = '',
		title,
		subtitle,
		center = false,
		level = 'h2'
	}: {
		badge?: string;
		title: string;
		subtitle: string;
		center?: boolean;
		level?: 'h1' | 'h2' | 'h3';
	} = $props();
</script>

<div class={`flex flex-col gap-4 ${center ? 'items-center text-center' : ''}`}>
	{#if badge}
		<div class="story-kicker">{badge}</div>
	{/if}
	<svelte:element this={level} class="page-title max-w-3xl">{title}</svelte:element>
	<p class={`soft-text max-w-2xl ${center ? 'text-center' : ''}`}>{subtitle}</p>
</div>
