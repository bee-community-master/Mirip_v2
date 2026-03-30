<script lang="ts">
	import { goto } from '$app/navigation';
	import { page } from '$app/state';
	import { ChevronLeft, ChevronRight, Heart, MessageCircle } from 'lucide-svelte';
	import GlassCard from '$lib/components/GlassCard.svelte';
	import Modal from '$lib/components/Modal.svelte';
	import SectionHeading from '$lib/components/SectionHeading.svelte';
	import { portfolioProfile, portfolioWorks } from '$lib/mocks/portfolio';

	const activeWork = $derived(
		portfolioWorks.find((work) => work.id === page.url.searchParams.get('work')) ?? null
	);
	const activeIndex = $derived(
		activeWork ? portfolioWorks.findIndex((work) => work.id === activeWork.id) : -1
	);

	function openWork(workId: string) {
		const params = new URLSearchParams(page.url.searchParams);
		params.set('work', workId);
		void goto(`${page.url.pathname}?${params.toString()}`, {
			replaceState: true,
			noScroll: true,
			keepFocus: true
		});
	}

	function closeWork() {
		const params = new URLSearchParams(page.url.searchParams);
		params.delete('work');
		const query = params.toString();
		void goto(query ? `${page.url.pathname}?${query}` : page.url.pathname, {
			replaceState: true,
			noScroll: true,
			keepFocus: true
		});
	}

	function move(delta: -1 | 1) {
		if (activeIndex < 0) return;
		const nextIndex = activeIndex + delta;
		if (nextIndex < 0 || nextIndex >= portfolioWorks.length) return;
		openWork(portfolioWorks[nextIndex].id);
	}
</script>

<svelte:head>
	<title>Portfolio | MIRIP v2</title>
	<meta
		name="description"
		content="프로필, 작업 아카이브, 대표작 상세 모달까지 한 흐름으로 보는 MIRIP 포트폴리오 시안입니다."
	/>
</svelte:head>

<section class="section-frame py-16 sm:py-20">
	<div class="space-y-12">
		<GlassCard className="overflow-hidden rounded-[34px] p-8 sm:p-10">
			<div class="grid gap-10 lg:grid-cols-[0.85fr_1.15fr] lg:items-end">
				<div class="max-w-xl">
					<SectionHeading
						badge="Profile"
						title={portfolioProfile.name}
						subtitle={portfolioProfile.role}
						level="h1"
					/>
				</div>

				<div class="flex flex-col gap-8 sm:flex-row sm:items-end sm:justify-between">
					<div class="flex items-center gap-5">
						<div class="rounded-full border border-fuchsia-400/30 p-1">
							<img
								src={portfolioProfile.avatar}
								alt={portfolioProfile.name}
								width="192"
								height="192"
								class="size-24 rounded-full object-cover"
								fetchpriority="high"
							/>
						</div>
						<p class="max-w-xs text-sm leading-7 font-medium text-white/58">
							진단 결과와 공모전 흐름을 함께 묶은 포트폴리오 프로필입니다. 각 작업은 모달에서 상세
							설명과 태그를 바로 확인할 수 있습니다.
						</p>
					</div>

					<div class="grid grid-cols-3 gap-4">
						{#each portfolioProfile.stats as stat}
							<div class="rounded-[24px] border border-white/8 bg-white/[0.03] px-5 py-4 text-center">
								<div class="font-display text-3xl font-black tracking-[-0.05em] text-white">
									{stat.value}
								</div>
								<div class="mt-1 text-xs font-bold uppercase tracking-[0.24em] text-white/35">
									{stat.label}
								</div>
							</div>
						{/each}
					</div>
				</div>
			</div>
		</GlassCard>

		<div class="column-masonry columns-1 md:columns-2 xl:columns-3">
			{#each portfolioWorks as work}
				<button
					type="button"
					class="group w-full rounded-[28px] focus-visible:outline focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-fuchsia-300"
					onclick={() => {
						openWork(work.id);
					}}
				>
					<GlassCard className="overflow-hidden rounded-[28px]">
						<div class={`${work.height === 'tall' ? 'aspect-[4/5]' : 'aspect-[4/4.2]'} relative overflow-hidden`}>
							<img
								src={work.image}
								alt={work.title}
								width="960"
								height={work.height === 'tall' ? '1200' : '1008'}
								class="h-full w-full object-cover transition-transform duration-500 group-hover:scale-[1.03]"
								loading="lazy"
							/>
							<div class="absolute inset-0 bg-gradient-to-t from-night-950 via-night-950/16 to-transparent opacity-90"></div>
							<div class="absolute inset-x-4 bottom-4">
								<div class="mb-2 flex flex-wrap gap-2">
									{#each work.tags.slice(0, 2) as tag}
										<span class="rounded-full border border-white/10 bg-black/30 px-2 py-1 text-[11px] font-semibold text-white/70">
											#{tag}
										</span>
									{/each}
								</div>
								<div class="flex items-end justify-between gap-4">
									<div class="min-w-0">
										<h2 class="font-display text-2xl font-bold tracking-[-0.04em] text-white">{work.title}</h2>
										<p class="mt-1 text-sm font-medium text-white/45">{work.year}</p>
									</div>
									<div class="text-right text-sm font-medium text-white/80">
										<div class="flex items-center justify-end gap-1">
											<Heart class="size-4" aria-hidden="true" />
											{work.likes}
										</div>
										<div class="mt-1 flex items-center justify-end gap-1 text-white/58">
											<MessageCircle class="size-4" aria-hidden="true" />
											{work.comments}
										</div>
									</div>
								</div>
							</div>
						</div>
					</GlassCard>
				</button>
			{/each}

			{#if !portfolioWorks.length}
				<GlassCard className="rounded-[28px] p-8 text-center">
					<h2 class="font-display text-2xl font-bold tracking-[-0.04em] text-white">
						아직 등록된 작업이 없습니다
					</h2>
					<p class="soft-text mt-3">
						포트폴리오 mock 데이터가 비어 있어도 화면이 깨지지 않도록 빈 상태를 먼저 보여줍니다.
					</p>
				</GlassCard>
			{/if}
		</div>
	</div>
</section>

<Modal
	open={Boolean(activeWork)}
	title={activeWork?.title ?? ''}
	onClose={closeWork}
	className="max-w-5xl"
>
	{#if activeWork}
		<div class="grid gap-6 lg:grid-cols-[1.15fr_0.85fr]">
			<div class="overflow-hidden rounded-[28px] border border-white/8">
				<img
					src={activeWork.image}
					alt={activeWork.title}
					width="1200"
					height="1500"
					class="h-full max-h-[70vh] w-full object-cover"
					loading="lazy"
				/>
			</div>

			<div class="flex min-w-0 flex-col gap-5">
				<div class="flex flex-wrap gap-2">
					{#each activeWork.tags as tag}
						<span class="rounded-full border border-white/10 bg-white/6 px-3 py-1 text-xs font-semibold text-white/70">
							#{tag}
						</span>
					{/each}
				</div>

				<p class="soft-text">{activeWork.description}</p>

				<div class="grid gap-3 sm:grid-cols-2">
					<div class="rounded-[24px] border border-white/8 bg-white/[0.03] p-4">
						<div class="mb-2 text-xs font-bold uppercase tracking-[0.24em] text-white/35">Likes</div>
						<div class="font-display text-3xl font-black tracking-[-0.05em] text-white">{activeWork.likes}</div>
					</div>
					<div class="rounded-[24px] border border-white/8 bg-white/[0.03] p-4">
						<div class="mb-2 text-xs font-bold uppercase tracking-[0.24em] text-white/35">Comments</div>
						<div class="font-display text-3xl font-black tracking-[-0.05em] text-white">{activeWork.comments}</div>
					</div>
				</div>

				<div class="mt-auto grid gap-3 sm:grid-cols-2">
					<button
						type="button"
						class="inline-flex items-center justify-center gap-2 rounded-full border border-white/12 bg-white/6 px-5 py-3 font-semibold text-white transition-colors duration-200 hover:bg-white/10 disabled:cursor-not-allowed disabled:opacity-40"
						onclick={() => {
							move(-1);
						}}
						disabled={activeIndex <= 0}
					>
						<ChevronLeft class="size-4" aria-hidden="true" />
						이전 작업
					</button>
					<button
						type="button"
						class="inline-flex items-center justify-center gap-2 rounded-full bg-white px-5 py-3 font-semibold text-black transition-transform duration-200 hover:scale-[1.02] disabled:cursor-not-allowed disabled:opacity-40"
						onclick={() => {
							move(1);
						}}
						disabled={activeIndex >= portfolioWorks.length - 1}
					>
						다음 작업
						<ChevronRight class="size-4" aria-hidden="true" />
					</button>
				</div>
			</div>
		</div>
	{/if}
</Modal>
