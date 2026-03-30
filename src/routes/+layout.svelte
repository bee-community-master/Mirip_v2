<script lang="ts">
	import './layout.css';
	import AppShell from '$lib/components/AppShell.svelte';
	import favicon from '$lib/assets/favicon.svg';

	let { children } = $props();
</script>

<svelte:head><link rel="icon" href={favicon} /></svelte:head>
<AppShell>{@render children()}</AppShell>
