<script lang="ts">
	import { goto } from '$app/navigation';
	import { page } from '$app/state';
	import {
		ArrowUpRight,
		Bookmark,
		Calendar,
		Filter,
		MapPin,
		Star
	} from 'lucide-svelte';
	import Drawer from '$lib/components/Drawer.svelte';
	import GlassCard from '$lib/components/GlassCard.svelte';
	import SectionHeading from '$lib/components/SectionHeading.svelte';
	import { competitionFilters, competitionItems } from '$lib/mocks/competitions';
	import type { CompetitionFilter } from '$lib/types';
	import { filterCompetitions } from '$lib/utils/competition';
	import { buildPathWithQuery, readQueryOption } from '$lib/utils/query';

	let bookmarkedIds = $state<string[]>([]);

	const selectedFilter = $derived(
		readQueryOption(page.url.searchParams, 'filter', competitionFilters, '전체')
	);

	const filteredCompetitions = $derived(filterCompetitions(competitionItems, selectedFilter));
	const selectedCompetition = $derived(
		filteredCompetitions.find((item) => item.id === page.url.searchParams.get('item')) ?? null
	);

	function updateQuery(next: { filter?: CompetitionFilter; item?: string | null }) {
		const nextPath = buildPathWithQuery(page.url.pathname, page.url.searchParams, {
			filter: next.filter === '전체' ? null : next.filter,
			item: next.item
		});

		void goto(nextPath, {
			replaceState: true,
			noScroll: true,
			keepFocus: true
		});
	}

	function selectFilter(filter: CompetitionFilter) {
		const currentItem = page.url.searchParams.get('item');
		const nextItems = filterCompetitions(competitionItems, filter);
		const nextSelected = nextItems.some((item) => item.id === currentItem) ? currentItem : null;

		updateQuery({ filter, item: nextSelected });
	}

	function openCompetition(itemId: string) {
		updateQuery({ item: itemId });
	}

	function closeDrawer() {
		updateQuery({ item: null });
	}

	function toggleBookmark(itemId: string) {
		bookmarkedIds = bookmarkedIds.includes(itemId)
			? bookmarkedIds.filter((id) => id !== itemId)
			: [...bookmarkedIds, itemId];
	}
</script>

<svelte:head>
	<title>Competitions | MIRIP v2</title>
	<meta
		name="description"
		content="작업 톤과 목표 대학에 맞는 공모전을 걸러 보고, 상세 패널에서 제출 포인트를 확인하세요."
	/>
</svelte:head>

<section class="section-frame py-16 sm:py-20">
	<div class="flex flex-col gap-10">
		<div class="flex flex-col gap-6 lg:flex-row lg:items-end lg:justify-between">
			<SectionHeading
				badge="Open Calls"
				title="진행중인 공모전"
				subtitle="필터를 바꾸면 현재 준비하는 작업 톤에 맞는 기회를 바로 좁혀서 볼 수 있습니다."
				level="h1"
			/>

			<div class="flex flex-wrap gap-2">
				{#each competitionFilters as filter}
					<button
						type="button"
						class={`rounded-full px-5 py-2.5 text-sm font-bold transition-colors duration-200 focus-visible:outline focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-fuchsia-300 ${
							selectedFilter === filter
								? 'bg-white text-black'
								: 'border border-white/10 bg-white/6 text-white/78 hover:bg-white/10'
						}`}
						aria-pressed={selectedFilter === filter}
						onclick={() => {
							selectFilter(filter);
						}}
					>
						<span class="inline-flex items-center gap-2">
							{#if filter === '전체'}
								<Filter class="size-4" aria-hidden="true" />
							{/if}
							{filter}
						</span>
					</button>
				{/each}
			</div>
		</div>

		<div class="grid gap-7 md:grid-cols-2 xl:grid-cols-3">
			{#each filteredCompetitions as competition}
				<GlassCard className="overflow-hidden rounded-[30px]" hoverable={true}>
					<button
						type="button"
						class="group h-full w-full rounded-[30px] text-left focus-visible:outline focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-fuchsia-300"
						onclick={() => {
							openCompetition(competition.id);
						}}
					>
						<div class="relative h-64 overflow-hidden">
							<img
								src={competition.image}
								alt={competition.title}
								width="960"
								height="640"
								class="h-full w-full object-cover transition-transform duration-500 group-hover:scale-[1.03]"
								loading="lazy"
							/>
							<div class="absolute inset-0 bg-gradient-to-t from-night-950 via-night-950/12 to-transparent"></div>
							<div class="absolute left-4 top-4 rounded-full bg-fuchsia-500 px-3 py-1 text-xs font-black text-white">
								{competition.dday}
							</div>
						</div>

						<div class="p-6">
							<div class="mb-4 flex flex-wrap gap-2">
								{#each competition.tags as tag}
									<span class="rounded-md border border-white/8 bg-white/6 px-2 py-1 text-xs font-medium text-white/72">
										#{tag}
									</span>
								{/each}
							</div>

							<div class="flex items-start justify-between gap-4">
								<div class="min-w-0">
									<h2 class="font-display text-2xl font-bold leading-tight tracking-[-0.04em] text-white text-balance">
										{competition.title}
									</h2>
									<p class="mt-2 text-sm font-medium text-white/52">{competition.org}</p>
								</div>
								<div class="rounded-full border border-white/8 bg-white/5 p-2 text-white/70">
									<ArrowUpRight class="size-4" aria-hidden="true" />
								</div>
							</div>

							<p class="soft-text mt-5 min-w-0">{competition.summary}</p>

							<div class="mt-6 flex items-center justify-between border-t border-white/8 pt-4 text-sm font-medium text-white/62">
								<span class="inline-flex items-center gap-2">
									<Calendar class="size-4 text-emerald-300" aria-hidden="true" />
									{competition.prize}
								</span>
								<span>{competition.deadline}</span>
							</div>
						</div>
					</button>
				</GlassCard>
			{/each}

			{#if !filteredCompetitions.length}
				<GlassCard className="rounded-[30px] p-8 md:col-span-2 xl:col-span-3">
					<h2 class="font-display text-2xl font-bold tracking-[-0.04em] text-white">
						조건에 맞는 공모전이 아직 없습니다
					</h2>
					<p class="soft-text mt-3">
						다른 카테고리 필터를 선택하거나 전체 보기로 돌아가면 추천 기회를 다시 볼 수 있습니다.
					</p>
					<button
						type="button"
						class="mt-6 rounded-full bg-white px-5 py-3 font-semibold text-black transition-transform duration-200 hover:scale-[1.02] focus-visible:outline focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-white"
						onclick={() => {
							selectFilter('전체');
						}}
					>
						전체 공모전 보기
					</button>
				</GlassCard>
			{/if}
		</div>
	</div>
</section>

<Drawer open={Boolean(selectedCompetition)} title={selectedCompetition?.title ?? ''} onClose={closeDrawer}>
	{#if selectedCompetition}
		<div class="space-y-6">
			<img
				src={selectedCompetition.image}
				alt={selectedCompetition.title}
				width="1200"
				height="900"
				class="h-[260px] w-full rounded-[28px] object-cover"
				loading="lazy"
			/>

			<div class="flex flex-wrap gap-2">
				{#each selectedCompetition.tags as tag}
					<span class="rounded-full border border-white/10 bg-white/6 px-3 py-1 text-xs font-semibold text-white/72">
						#{tag}
					</span>
				{/each}
			</div>

			<p class="soft-text">{selectedCompetition.summary}</p>

			<div class="grid gap-3 sm:grid-cols-2">
				<div class="rounded-[24px] border border-white/8 bg-white/[0.03] p-4">
					<div class="mb-2 text-xs font-bold uppercase tracking-[0.24em] text-white/35">Deadline</div>
					<div class="text-lg font-semibold text-white">{selectedCompetition.deadline}</div>
				</div>
				<div class="rounded-[24px] border border-white/8 bg-white/[0.03] p-4">
					<div class="mb-2 text-xs font-bold uppercase tracking-[0.24em] text-white/35">Region</div>
					<div class="inline-flex items-center gap-2 text-lg font-semibold text-white">
						<MapPin class="size-4 text-white/45" aria-hidden="true" />
						{selectedCompetition.region}
					</div>
				</div>
				<div class="rounded-[24px] border border-white/8 bg-white/[0.03] p-4">
					<div class="mb-2 text-xs font-bold uppercase tracking-[0.24em] text-white/35">Participants</div>
					<div class="text-lg font-semibold text-white">{selectedCompetition.participants}</div>
				</div>
				<div class="rounded-[24px] border border-white/8 bg-white/[0.03] p-4">
					<div class="mb-2 text-xs font-bold uppercase tracking-[0.24em] text-white/35">Difficulty</div>
					<div class="text-lg font-semibold text-white">{selectedCompetition.difficulty}</div>
				</div>
			</div>

			<div class="rounded-[28px] border border-white/8 bg-white/[0.03] p-5">
				<div class="mb-4 inline-flex items-center gap-2 text-sm font-bold text-fuchsia-200">
					<Star class="size-4" aria-hidden="true" />
					왜 이 공모전이 잘 맞는지
				</div>
				<ul class="space-y-3">
					{#each selectedCompetition.matchingReason as reason}
						<li class="soft-text flex gap-3">
							<span class="mt-2 size-1.5 rounded-full bg-fuchsia-300" aria-hidden="true"></span>
							<span>{reason}</span>
						</li>
					{/each}
				</ul>
			</div>

			<div class="grid gap-3 sm:grid-cols-2">
				<button
					type="button"
					class="rounded-full bg-white px-5 py-3 font-semibold text-black transition-transform duration-200 hover:scale-[1.02]"
					onclick={() => {
						toggleBookmark(selectedCompetition.id);
					}}
				>
					{bookmarkedIds.includes(selectedCompetition.id) ? '관심 저장됨' : '관심 공모전 저장'}
				</button>
				<button
					type="button"
					class="rounded-full border border-white/12 bg-white/6 px-5 py-3 font-semibold text-white transition-colors duration-200 hover:bg-white/10"
					onclick={closeDrawer}
				>
					닫기
				</button>
			</div>
		</div>
	{/if}
</Drawer>
