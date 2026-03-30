<script lang="ts">
	let {
		badge = '',
		title,
		subtitle,
		center = false
	}: {
		badge?: string;
		title: string;
		subtitle: string;
		center?: boolean;
	} = $props();
</script>

<div class={`flex flex-col gap-4 ${center ? 'items-center text-center' : ''}`}>
	{#if badge}
		<div class="story-kicker">{badge}</div>
	{/if}
	<h2 class="page-title max-w-3xl">{title}</h2>
	<p class={`soft-text max-w-2xl ${center ? 'text-center' : ''}`}>{subtitle}</p>
</div>
