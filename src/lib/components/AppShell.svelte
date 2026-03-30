<script lang="ts">
	import type { Snippet } from 'svelte';
	import { page } from '$app/state';
	import {
		Menu,
		Palette,
		Sparkles,
		UploadCloud,
		UserRound,
		X
	} from 'lucide-svelte';
	import MockAuthModal from '$lib/components/MockAuthModal.svelte';

	let { children }: { children?: Snippet } = $props();

	let mobileOpen = $state(false);
	let authOpen = $state(false);
	let scrolled = $state(false);

	const pathname = $derived(page.url.pathname);

	const navItems = [
		{ name: 'Home', path: '/', icon: Sparkles },
		{ name: 'AI Diagnosis', path: '/diagnosis', icon: UploadCloud },
		{ name: 'Competitions', path: '/competitions', icon: Palette },
		{ name: 'Portfolio', path: '/portfolio', icon: UserRound }
	];

	function isActive(path: string) {
		return path === '/' ? pathname === '/' : pathname.startsWith(path);
	}

	function closeMobileMenu() {
		mobileOpen = false;
	}
</script>

<svelte:window
	onscroll={() => {
		scrolled = window.scrollY > 18;
	}}
/>

<div class="relative min-h-screen overflow-x-hidden">
	<div class="pointer-events-none fixed inset-0 -z-10 overflow-hidden">
		<div
			class="animate-drift absolute left-[-12vw] top-[-14vw] h-[48vw] w-[48vw] rounded-full bg-fuchsia-500/18 blur-[140px]"
		></div>
		<div
			class="animate-drift absolute bottom-[-16vw] right-[-10vw] h-[54vw] w-[54vw] rounded-full bg-blue-500/18 blur-[160px]"
			style="animation-delay: -6s"
		></div>
		<div
			class="absolute inset-x-0 top-0 h-[620px] bg-[radial-gradient(circle_at_top,rgba(255,255,255,0.06),transparent_50%)]"
		></div>
	</div>

	<header
		class={`fixed inset-x-0 top-0 z-50 transition duration-300 ${scrolled ? 'border-b border-white/8 bg-night-950/70 shadow-[0_18px_50px_rgba(0,0,0,0.25)] backdrop-blur-xl' : 'bg-transparent'}`}
	>
		<a
			href="#main-content"
			class="sr-only absolute left-4 top-3 rounded-full bg-white px-4 py-2 text-sm font-semibold text-black focus:not-sr-only focus:z-50"
		>
			본문으로 건너뛰기
		</a>
		<div class="section-frame flex h-16 items-center justify-between gap-4">
			<a
				href="/"
				class="font-display rounded-full text-2xl font-black tracking-[-0.08em] text-gradient focus-visible:outline focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-fuchsia-300"
			>
				MIRIP.
			</a>

			<nav class="hidden items-center gap-2 md:flex">
				{#each navItems as item}
					<a
						href={item.path}
						class={`relative rounded-full px-4 py-2 text-sm font-semibold transition focus-visible:outline focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-fuchsia-300 ${isActive(item.path) ? 'text-white' : 'text-white/52 hover:text-white'}`}
						aria-current={isActive(item.path) ? 'page' : undefined}
					>
						{#if isActive(item.path)}
							<span class="absolute inset-0 rounded-full bg-white/10"></span>
						{/if}
						<span class="relative z-10 flex items-center gap-2">
							<item.icon class="size-4" aria-hidden="true" />
							{item.name}
						</span>
					</a>
				{/each}
			</nav>

			<div class="flex items-center gap-3">
				<button
					type="button"
					class="hidden rounded-full bg-white px-5 py-2 text-sm font-semibold text-black transition hover:scale-[1.03] focus-visible:outline focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-white md:block"
					onclick={() => {
						authOpen = true;
					}}
				>
					Sign In
				</button>

				<button
					type="button"
					class="inline-flex rounded-full border border-white/10 bg-white/6 p-2 text-white focus-visible:outline focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-fuchsia-300 md:hidden"
					onclick={() => {
						mobileOpen = !mobileOpen;
					}}
					aria-label={mobileOpen ? '메뉴 닫기' : '메뉴 열기'}
					aria-expanded={mobileOpen}
					aria-controls="mobile-navigation"
				>
					{#if mobileOpen}
						<X class="size-5" aria-hidden="true" />
					{:else}
						<Menu class="size-5" aria-hidden="true" />
					{/if}
				</button>
			</div>
		</div>

		{#if mobileOpen}
			<div
				id="mobile-navigation"
				class="border-t border-white/8 bg-night-950/92 px-4 py-4 backdrop-blur-xl md:hidden"
			>
				<div class="flex flex-col gap-2">
					{#each navItems as item}
						<a
							href={item.path}
							class={`rounded-2xl px-4 py-3 text-sm font-semibold focus-visible:outline focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-fuchsia-300 ${isActive(item.path) ? 'bg-white text-black' : 'bg-white/5 text-white/70'}`}
							aria-current={isActive(item.path) ? 'page' : undefined}
							onclick={closeMobileMenu}
						>
							<span class="flex items-center gap-2">
								<item.icon class="size-4" aria-hidden="true" />
								{item.name}
							</span>
						</a>
					{/each}
					<button
						type="button"
						class="mt-3 rounded-2xl bg-white px-4 py-3 text-sm font-semibold text-black focus-visible:outline focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-white"
						onclick={() => {
							closeMobileMenu();
							authOpen = true;
						}}
					>
						Sign In
					</button>
				</div>
			</div>
		{/if}
	</header>

	<main id="main-content" class="relative pt-16">
		{@render children?.()}
	</main>

	<footer class="relative mt-24 border-t border-white/8 bg-night-950/70 py-10 backdrop-blur-xl">
		<div
			class="section-frame flex flex-col gap-3 text-sm text-white/45 sm:flex-row sm:items-center sm:justify-between"
		>
			<div>
				<div class="font-display text-xl font-bold tracking-[-0.05em] text-white/80">MIRIP AI Diagnostic Service</div>
				<p class="mt-1">AI diagnosis, competitions, and portfolio flow in one frontend prototype.</p>
			</div>
			<p>© 2026 Chanspick. All rights reserved.</p>
		</div>
	</footer>

	<MockAuthModal
		open={authOpen}
		onClose={() => {
			authOpen = false;
		}}
	/>
</div>
