<script lang="ts">
	import { ArrowRight, Sparkles, TrendingUp, Trophy } from 'lucide-svelte';
	import GlassCard from '$lib/components/GlassCard.svelte';
	import { homeFeatures, homeHeroImages } from '$lib/mocks/home';
</script>

<svelte:head>
	<title>MIRIP v2 | AI 미술 입시 진단</title>
	<meta
		name="description"
		content="내 그림이 어디까지 갈 수 있는지, AI 진단과 공모전 매칭, 포트폴리오 흐름으로 한 번에 확인하세요."
	/>
</svelte:head>

<section class="relative flex min-h-[calc(100svh-4rem)] items-center overflow-hidden py-12 lg:py-0">
	<div class="section-frame grid gap-14 lg:grid-cols-[0.94fr_1.06fr] lg:items-center">
		<div class="max-w-2xl animate-rise">
			<div class="story-kicker">
				<Sparkles class="size-4" aria-hidden="true" />
				AI 기반 미술 입시 진단 V2.0
			</div>

			<h1 class="hero-headline mt-6 text-balance">
				내 그림,
				<br />
				어느 대학까지
				<br />
				<span class="text-gradient">갈 수 있을까?</span>
			</h1>

			<p class="soft-text mt-6 max-w-xl text-base sm:text-lg">
				최신 DINOv2 기반 분석과 포트폴리오 맥락을 조합해 현재 위치와 다음 행동을 보여줍니다.
				진단, 공모전, 작업 기록을 같은 흐름 안에서 이어갈 수 있게 구성했습니다.
			</p>

			<div class="mt-10 flex flex-col gap-4 sm:flex-row">
				<a
					href="/diagnosis"
					class="inline-flex items-center justify-center gap-2 rounded-full bg-white px-7 py-4 text-base font-bold text-black transition-transform duration-200 hover:scale-[1.02] focus-visible:outline focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-white"
				>
					AI 무료 진단받기
					<ArrowRight class="size-5" aria-hidden="true" />
				</a>
				<a
					href="/competitions"
					class="inline-flex items-center justify-center gap-2 rounded-full border border-white/12 bg-white/6 px-7 py-4 text-base font-bold text-white transition-colors duration-200 hover:bg-white/10 focus-visible:outline focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-fuchsia-300"
				>
					진행중인 공모전 보기
				</a>
			</div>
		</div>

		<div class="relative hidden h-[620px] lg:block">
			{#each homeHeroImages as image, index}
				<figure
					class={`animate-float-card absolute overflow-hidden rounded-[32px] border border-white/14 bg-night-900 shadow-[0_18px_80px_rgba(0,0,0,0.45)] ${
						index === 0
							? 'right-8 top-4 z-30 h-[430px] w-[310px]'
							: index === 1
								? 'bottom-8 right-40 z-20 h-[320px] w-[250px]'
								: 'left-4 top-28 z-10 h-[360px] w-[280px]'
					}`}
					style={`animation-delay: ${index * 0.55}s;`}
				>
					<img
						src={image}
						alt="MIRIP artwork preview"
						width="620"
						height="840"
						class="h-full w-full object-cover"
						loading={index === 0 ? 'eager' : 'lazy'}
						fetchpriority={index === 0 ? 'high' : undefined}
					/>
					<div class="absolute inset-0 bg-gradient-to-t from-night-950 via-night-950/10 to-transparent"></div>
					<div class="absolute inset-x-4 bottom-4 rounded-2xl border border-white/10 bg-black/30 p-3 backdrop-blur-md">
						<div class="mb-2 h-2 overflow-hidden rounded-full bg-white/10">
							<div
								class="h-full rounded-full bg-gradient-to-r from-fuchsia-400 to-azure-450"
								style={`width: ${index === 0 ? 95 : index === 1 ? 82 : 70}%`}
							></div>
						</div>
						<div class="flex items-center justify-between text-xs font-bold text-white/75">
							<span>AI 매칭률</span>
							<span>{index === 0 ? 95 : index === 1 ? 82 : 70}%</span>
						</div>
					</div>
				</figure>
			{/each}
		</div>
	</div>
</section>

<section class="border-y border-white/6 py-20">
	<div class="section-frame grid gap-6 md:grid-cols-3">
		{#each homeFeatures as feature}
			<GlassCard className="rounded-[30px] p-8" hoverable={true}>
				<div class="mb-6 flex size-14 items-center justify-center rounded-2xl bg-gradient-to-br from-fuchsia-500 to-azure-450 text-white">
					{#if feature.icon === 'trend'}
						<TrendingUp class="size-7" aria-hidden="true" />
					{:else if feature.icon === 'sparkles'}
						<Sparkles class="size-7" aria-hidden="true" />
					{:else}
						<Trophy class="size-7" aria-hidden="true" />
					{/if}
				</div>
				<h2 class="font-display text-2xl font-bold tracking-[-0.04em] text-white">{feature.title}</h2>
				<p class="soft-text mt-3">{feature.description}</p>
			</GlassCard>
		{/each}
	</div>
</section>
