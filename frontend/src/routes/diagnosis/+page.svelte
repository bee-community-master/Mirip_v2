<script lang="ts">
	import { goto } from '$app/navigation';
	import { page } from '$app/state';
	import { onDestroy } from 'svelte';
	import {
		AlertCircle,
		BookOpen,
		CheckCircle2,
		GraduationCap,
		Lock,
		Sparkles,
		TrendingUp,
		Upload
	} from 'lucide-svelte';
	import GlassCard from '$lib/components/GlassCard.svelte';
	import RadarChart from '$lib/components/RadarChart.svelte';
	import SectionHeading from '$lib/components/SectionHeading.svelte';
	import { diagnosisMock } from '$lib/mocks/diagnosis';
	import type { ExpertTab, TierKey, UniversityKey } from '$lib/types';
	import { buildPathWithQuery, readQueryOption } from '$lib/utils/query';

	const tierOptions = ['FREE', 'STANDARD', 'PRO'] as const;
	const universityOptions = ['HONGIK', 'KONKUK', 'KOOKMIN'] as const;
	const expertOptions = ['AI', 'INSTRUCTOR', 'PROFESSOR'] as const;
	const positioningGrades = ['S', 'A', 'B', 'C'] as const;
	const validImageTypes = ['image/png', 'image/jpeg', 'image/jpg', 'image/webp'];
	const maxUploadSizeInBytes = 10 * 1024 * 1024;
	const expertGuideTabs = [
		{ id: 'AI', icon: Sparkles, label: 'AI 분석' },
		{ id: 'INSTRUCTOR', icon: BookOpen, label: '입시 강사 의견' },
		{ id: 'PROFESSOR', icon: GraduationCap, label: '교수 의견' }
	] as const satisfies ReadonlyArray<{
		id: ExpertTab;
		icon: typeof Sparkles;
		label: string;
	}>;
	const universityTabs = universityOptions.map((key) => ({
		key,
		label: diagnosisMock.universities[key].label
	})) satisfies ReadonlyArray<{
		key: UniversityKey;
		label: string;
	}>;
	const tierOrder: Record<TierKey, number> = {
		FREE: 0,
		STANDARD: 1,
		PRO: 2
	};

	type Stage = 'upload' | 'analyzing' | 'result';

	let stage = $state<Stage>('upload');
	let selectedFile = $state<File | null>(null);
	let previewUrl = $state<string | null>(null);
	let error = $state('');
	let isDragging = $state(false);
	let analyzingTimer: ReturnType<typeof setTimeout> | null = null;

	const tier = $derived(readQueryOption(page.url.searchParams, 'tier', tierOptions, 'FREE'));
	const universityKey = $derived(
		readQueryOption(page.url.searchParams, 'uni', universityOptions, 'HONGIK')
	);
	const expertTab = $derived(
		readQueryOption(page.url.searchParams, 'expert', expertOptions, 'AI')
	);

	const selectedUniversity = $derived(diagnosisMock.universities[universityKey]);
	const standardUnlocked = $derived(tierOrder[tier] >= tierOrder.STANDARD);
	const proUnlocked = $derived(tierOrder[tier] >= tierOrder.PRO);

	function updateQuery(next: { tier?: TierKey; uni?: UniversityKey; expert?: ExpertTab }) {
		const nextPath = buildPathWithQuery(page.url.pathname, page.url.searchParams, {
			tier: next.tier === 'FREE' ? null : next.tier,
			uni: next.uni === 'HONGIK' ? null : next.uni,
			expert: next.expert === 'AI' ? null : next.expert
		});

		void goto(nextPath, {
			replaceState: true,
			noScroll: true,
			keepFocus: true
		});
	}

	function clearPreview() {
		if (previewUrl) {
			URL.revokeObjectURL(previewUrl);
			previewUrl = null;
		}
	}

	function selectTier(nextTier: TierKey) {
		updateQuery({ tier: nextTier });
	}

	function selectUniversity(nextUniversity: UniversityKey) {
		updateQuery({ uni: nextUniversity });
	}

	function selectExpertGuide(nextExpert: ExpertTab) {
		updateQuery({ expert: nextExpert });
	}

	function clearTimer() {
		if (analyzingTimer) {
			clearTimeout(analyzingTimer);
			analyzingTimer = null;
		}
	}

	function beginAnalysis(file: File) {
		if (!validImageTypes.includes(file.type)) {
			error = 'PNG, JPG, JPEG, WebP 형식의 이미지만 업로드할 수 있습니다.';
			return;
		}

		if (file.size > maxUploadSizeInBytes) {
			error = '파일 크기는 10 MB 이하로 올려주세요.';
			return;
		}

		clearTimer();
		clearPreview();

		error = '';
		selectedFile = file;
		previewUrl = URL.createObjectURL(file);
		stage = 'analyzing';

		analyzingTimer = setTimeout(() => {
			stage = 'result';
		}, 2200);
	}

	function handleInputChange(event: Event) {
		const target = event.currentTarget as HTMLInputElement;
		const file = target.files?.[0];
		if (file) beginAnalysis(file);
		target.value = '';
	}

	function handleDrop(event: DragEvent) {
		event.preventDefault();
		isDragging = false;

		const file = event.dataTransfer?.files?.[0];
		if (file) beginAnalysis(file);
	}

	function resetDiagnosis() {
		clearTimer();
		clearPreview();
		error = '';
		selectedFile = null;
		stage = 'upload';
		updateQuery({ tier: 'FREE', uni: 'HONGIK', expert: 'AI' });
	}

	onDestroy(() => {
		clearTimer();
		clearPreview();
	});
</script>

<svelte:head>
	<title>Diagnosis | MIRIP v2</title>
	<meta
		name="description"
		content="업로드부터 결과, 대학 탭, 전문가 탭, 잠금 해제 상태까지 mock 데이터로 동작하는 MIRIP AI 진단 화면입니다."
	/>
</svelte:head>

<section class="section-frame py-16 sm:py-20">
	<div class="mx-auto flex max-w-6xl flex-col gap-10">
		<SectionHeading
			badge="AI Diagnosis"
			title="작품을 업로드하고 대학별 합격 가능성을 확인하세요"
			subtitle="실제 API 없이도 업로드, 분석 상태, 결과, 티어 전환, 대학/전문가 탭을 모두 로컬 상태와 fixture로 재현합니다."
			center={true}
			level="h1"
		/>

		{#if error}
			<div
				class="mx-auto flex w-full max-w-2xl items-start gap-3 rounded-[24px] border border-rose-400/20 bg-rose-500/10 px-5 py-4 text-sm font-medium text-rose-100"
				aria-live="polite"
			>
				<AlertCircle class="mt-0.5 size-4 shrink-0" aria-hidden="true" />
				<span>{error}</span>
			</div>
		{/if}

		{#if stage === 'upload'}
			<label
				class={`mx-auto flex h-80 w-full max-w-2xl cursor-pointer flex-col items-center justify-center rounded-[36px] border-2 border-dashed px-8 text-center transition-colors duration-200 focus-within:border-fuchsia-400 focus-within:bg-white/8 ${isDragging ? 'border-fuchsia-400 bg-fuchsia-500/10' : 'border-white/16 bg-white/5 hover:border-fuchsia-400/50 hover:bg-white/8'}`}
				ondragover={(event) => {
					event.preventDefault();
					isDragging = true;
				}}
				ondragleave={(event) => {
					event.preventDefault();
					isDragging = false;
				}}
				ondrop={handleDrop}
			>
				<input
					type="file"
					class="sr-only"
					accept="image/png,image/jpeg,image/jpg,image/webp"
					name="artwork_upload"
					onchange={handleInputChange}
				/>
				<div class="mb-6 flex size-20 items-center justify-center rounded-full border border-white/10 bg-night-900 text-fuchsia-300 shadow-[0_24px_80px_rgba(6,8,20,0.35)]">
					<Upload class="size-8" aria-hidden="true" />
				</div>
				<h2 class="font-display text-3xl font-bold tracking-[-0.04em] text-white">
					클릭하거나 이미지를 드래그하세요
				</h2>
				<p class="soft-text mt-3 max-w-md">PNG, JPG, JPEG, WebP, 최대 10 MB까지 지원합니다.</p>
			</label>
		{:else if stage === 'analyzing'}
			<div class="mx-auto flex flex-col items-center justify-center py-16" aria-live="polite">
				<div class="relative mb-8 size-32">
					<div class="absolute inset-0 rounded-full border-4 border-white/10"></div>
					<div
						class="absolute inset-0 rounded-full border-4 border-fuchsia-500 border-t-transparent"
						style="animation: spin 1.8s linear infinite;"
					></div>
					<div class="absolute inset-0 flex items-center justify-center">
						<Sparkles class="size-8 text-fuchsia-300" aria-hidden="true" />
					</div>
				</div>

				<h2 class="font-display text-3xl font-bold tracking-[-0.05em] text-white">
					DINOv2 AI 분석 중…
				</h2>
				<p class="soft-text mt-3">
					수만 개의 합격작 데이터와 업로드한 작품을 대조하고 있습니다.
				</p>
			</div>
		{:else}
			<div class="flex flex-col gap-12">
				<div class="sticky top-20 z-30 flex justify-center">
					<div class="glass-panel no-scrollbar flex items-center gap-2 overflow-x-auto rounded-full px-3 py-2">
						<span class="px-3 text-[11px] font-black uppercase tracking-[0.24em] text-white/35">Demo</span>
						{#each tierOptions as option}
							<button
								type="button"
								class={`rounded-full px-4 py-2 text-sm font-bold transition-colors duration-200 focus-visible:outline focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-fuchsia-300 ${
									tier === option ? 'bg-white text-black' : 'text-white/52 hover:bg-white/6 hover:text-white'
								}`}
								aria-pressed={tier === option}
								onclick={() => {
									selectTier(option);
								}}
							>
								{option}
							</button>
						{/each}
					</div>
				</div>

				<section class="space-y-6">
					<div class="flex flex-wrap items-center gap-3">
						<span class="rounded-full border border-white/10 bg-white/6 px-3 py-1 text-xs font-black text-white/75">FREE</span>
						<h2 class="font-display text-3xl font-black tracking-[-0.05em] text-white">작품 진단 결과</h2>
					</div>

					<div class="grid gap-6 lg:grid-cols-[1.05fr_0.95fr]">
						<GlassCard className="rounded-[30px] p-7 sm:p-8">
							<div class="grid gap-6 xl:grid-cols-[1fr_240px] xl:items-center">
								<div class="flex flex-col gap-5">
									<div class="h-[320px]">
										<RadarChart data={diagnosisMock.radar} />
									</div>
									<div>
										<div class="flex h-8 overflow-hidden rounded-full text-[11px] font-black">
											{#each diagnosisMock.tierResult.segments as segment}
												<div
													class={`flex items-center justify-center ${segment.fillClass} ${segment.textClass}`}
													style={`flex: ${segment.value};`}
												>
													{segment.label}
												</div>
											{/each}
										</div>
										<p class="mt-4 text-center text-sm font-medium text-white/60">
											<span class="font-bold text-white">{diagnosisMock.tierResult.predictedGrade}등급</span>
											확률 {diagnosisMock.tierResult.probability}% · 신뢰구간 ±1등급 내 적중률
											<span class="font-bold text-white">{diagnosisMock.tierResult.confidence}%</span>
										</p>
									</div>
								</div>

								<div class="rounded-[28px] border border-white/8 bg-night-900/70 p-5">
									<div class="mb-4 flex items-center justify-between">
										<h3 class="font-display text-2xl font-bold tracking-[-0.04em] text-white">
											Uploaded
										</h3>
										<span class="rounded-full bg-fuchsia-500/16 px-3 py-1 text-xs font-bold text-fuchsia-200">
											{selectedFile?.name ? 'Mock Ready' : 'Sample'}
										</span>
									</div>
									<div class="overflow-hidden rounded-[24px] border border-white/8">
										<img
											src={previewUrl ?? 'https://images.unsplash.com/photo-1593472807861-5bb884af28f6?q=80&w=800&auto=format&fit=crop'}
											alt="업로드한 작품 미리보기"
											width="900"
											height="1100"
											class="aspect-[4/5] w-full object-cover"
										/>
									</div>
									<p class="soft-text mt-4">
										현재 업로드한 이미지를 기준으로 분석한 mock 결과입니다. 파일을 다시 올리면 상태가 처음부터 다시 시작됩니다.
									</p>
								</div>
							</div>
						</GlassCard>

						<GlassCard className="rounded-[30px] p-7 sm:p-8">
							<h3 class="font-display text-2xl font-bold tracking-[-0.04em] text-white">전체 포지셔닝</h3>
							<div class="relative mt-6 min-h-[320px] overflow-hidden rounded-[28px] border border-white/8 bg-night-900/70">
								<div class="absolute left-1/2 top-10 bottom-10 w-px -translate-x-1/2 bg-gradient-to-b from-gold-350 via-white/60 to-azure-450"></div>
								{#each positioningGrades as grade, index}
									<div
										class="absolute left-1/2 -translate-x-1/2"
										style={`top: ${12 + index * 26}%`}
									>
										<div
											class={`flex size-10 items-center justify-center rounded-full font-black ${
												grade === 'S'
													? 'bg-yellow-400 text-black'
													: grade === 'A'
														? 'bg-white text-black'
														: grade === 'B'
															? 'bg-orange-500 text-white'
															: 'bg-blue-600 text-white'
											}`}
										>
											{grade}
										</div>
									</div>
								{/each}

								<div class="absolute left-1/2 top-[44%] ml-8 flex items-center gap-3">
									<div class="relative flex size-6 items-center justify-center">
										<div class="absolute size-6 rounded-full bg-fuchsia-400/45 animate-pulse-ring"></div>
										<div class="size-3 rounded-full bg-fuchsia-400"></div>
									</div>
									<span class="rounded bg-fuchsia-500 px-2 py-1 text-xs font-black text-white">YOU</span>
								</div>
							</div>
						</GlassCard>
					</div>
				</section>

				<section class="relative space-y-6 border-t border-white/8 pt-10">
					<div class="flex flex-wrap items-center gap-3">
						<span class="rounded-full border border-yellow-400/20 bg-yellow-500/10 px-3 py-1 text-xs font-black text-yellow-300">
							STANDARD
						</span>
						<span class="text-sm font-bold text-white/42">$20 / 1회</span>
					</div>
					<h2 class="font-display text-3xl font-black tracking-[-0.05em] text-white">가장 가까운 대학은?</h2>

					<div class="relative">
						<div
							class={`grid gap-6 md:grid-cols-3 ${!standardUnlocked ? 'pointer-events-none select-none blur-sm opacity-35' : ''}`}
						>
							{#each diagnosisMock.standardMatches as match}
								<GlassCard className="rounded-[28px] p-6">
									<div class="flex items-start justify-between gap-4">
										<h3 class="min-w-0 font-display text-2xl font-bold leading-tight tracking-[-0.04em] text-white">
											{match.name}
										</h3>
										<span class="font-display text-3xl font-black tracking-[-0.05em] text-fuchsia-300">
											{match.match}%
										</span>
									</div>
									<div class="mt-6 space-y-2 text-sm font-medium text-white/58">
										<div class="flex justify-between"><span>지원자</span><span class="text-white">{match.applicants}</span></div>
										<div class="flex justify-between"><span>경쟁률</span><span class="text-white">{match.ratio}</span></div>
										<div class="flex justify-between"><span>실기 예상</span><span class="text-white">상위 {match.rank}</span></div>
									</div>
								</GlassCard>
							{/each}
						</div>

						{#if !standardUnlocked}
							<div class="absolute inset-0 z-10 flex items-center justify-center">
								<div class="glass-panel rounded-[32px] px-7 py-6 text-center">
									<Lock class="mx-auto size-7 text-white/60" aria-hidden="true" />
									<p class="mt-4 text-base font-semibold text-white">
										Standard 이상에서 대학 매칭을 확인할 수 있습니다.
									</p>
									<button
										type="button"
										class="mt-5 rounded-full bg-white px-6 py-3 font-semibold text-black transition-transform duration-200 hover:scale-[1.02]"
										onclick={() => {
											updateQuery({ tier: 'STANDARD' });
										}}
									>
										대학 매칭 결과 열기 — $20
									</button>
								</div>
							</div>
						{/if}
					</div>
				</section>

				<section class="relative space-y-6 border-t border-white/8 pt-10">
					<div class="flex flex-wrap items-center gap-3">
						<span class="rounded-full bg-gradient-to-r from-fuchsia-500 to-azure-450 px-3 py-1 text-xs font-black text-white">
							PRO
						</span>
						<span class="text-sm font-bold text-white/42">+$40 (총 $60)</span>
					</div>
					<h2 class="font-display text-3xl font-black tracking-[-0.05em] text-white">대학별 맞춤 분석</h2>

					<div class="relative">
						<GlassCard className={`overflow-hidden rounded-[32px] ${!proUnlocked ? 'pointer-events-none select-none blur-md opacity-25' : ''}`}>
							<div class="no-scrollbar flex overflow-x-auto border-b border-white/8">
								{#each universityTabs as universityTab}
									<button
										type="button"
										class={`border-b-2 px-8 py-4 text-sm font-bold whitespace-nowrap transition-colors duration-200 ${
											universityKey === universityTab.key
												? 'border-fuchsia-400 bg-white/6 text-white'
												: 'border-transparent text-white/45 hover:text-white'
										}`}
										aria-pressed={universityKey === universityTab.key}
										onclick={() => {
											selectUniversity(universityTab.key);
										}}
									>
										{universityTab.label}
									</button>
								{/each}
							</div>

							<div class="space-y-10 p-6 sm:p-8">
								<div class="grid gap-4 sm:grid-cols-3">
									{#each selectedUniversity.stats as stat}
										<div class={`rounded-[22px] p-4 text-center ${stat.accent ? 'border border-fuchsia-400/20 bg-fuchsia-500/10' : 'border border-white/8 bg-white/[0.03]'}`}>
											<div class={`text-xs font-bold ${stat.accent ? 'text-fuchsia-200' : 'text-white/42'}`}>
												{stat.label}
											</div>
											<div class={`mt-1 font-display text-2xl font-black tracking-[-0.05em] ${stat.accent ? 'text-fuchsia-200' : 'text-white'}`}>
												{stat.value}
											</div>
										</div>
									{/each}
								</div>

								<div>
									<h3 class="font-display text-2xl font-bold tracking-[-0.04em] text-white">축별 합격 거리</h3>
									<div class="mt-4 grid gap-4 md:grid-cols-4">
										{#each selectedUniversity.axisDistances as axis}
											<div class="rounded-[22px] border border-white/8 bg-night-900/70 p-4 text-center">
												<div class="text-sm font-bold text-white/55">{axis.label}</div>
												<div
													class={`mt-3 inline-flex rounded-full px-3 py-1 text-sm font-black ${
														axis.needImprovement
															? 'bg-orange-500/16 text-orange-200'
															: 'bg-emerald-500/14 text-emerald-200'
													}`}
												>
													{axis.value}
												</div>
											</div>
										{/each}
									</div>
								</div>

								<div class="grid gap-6 md:grid-cols-2">
									<div class="rounded-[24px] border border-blue-400/20 bg-blue-500/10 p-6">
										<div class="mb-4 flex items-center gap-2 text-lg font-black text-blue-200">
											<CheckCircle2 class="size-5" aria-hidden="true" />
											강점
										</div>
										<ul class="space-y-2">
											{#each selectedUniversity.strengths as item}
												<li class="soft-text flex gap-3">
													<span class="mt-2 size-1.5 rounded-full bg-blue-200" aria-hidden="true"></span>
													<span>{item}</span>
												</li>
											{/each}
										</ul>
									</div>

									<div class="rounded-[24px] border border-orange-400/20 bg-orange-500/10 p-6">
										<div class="mb-4 flex items-center gap-2 text-lg font-black text-orange-200">
											<AlertCircle class="size-5" aria-hidden="true" />
											보완점
										</div>
										<ul class="space-y-2">
											{#each selectedUniversity.improvements as item}
												<li class="soft-text flex gap-3">
													<span class="mt-2 size-1.5 rounded-full bg-orange-200" aria-hidden="true"></span>
													<span>{item}</span>
												</li>
											{/each}
										</ul>
									</div>
								</div>

								<div>
									<h3 class="inline-flex items-center gap-2 font-display text-2xl font-bold tracking-[-0.04em] text-white">
										<TrendingUp class="size-5 text-fuchsia-300" aria-hidden="true" />
										역대 기조 대비
									</h3>

									<div class="mt-4 space-y-4">
										{#each selectedUniversity.historyNotes as note}
											<div
												class={`rounded-r-[22px] border-l-2 p-4 ${
													note.tone === 'blue'
														? 'border-blue-400 bg-blue-500/10'
														: note.tone === 'gold'
															? 'border-yellow-400 bg-yellow-500/10'
															: 'border-rose-400 bg-rose-500/10'
												}`}
											>
												<div class={`text-xs font-black ${note.tone === 'blue' ? 'text-blue-200' : note.tone === 'gold' ? 'text-yellow-200' : 'text-rose-200'}`}>
													{note.label}
												</div>
												<p class="mt-2 text-sm font-medium leading-7 text-white/70">{note.text}</p>
											</div>
										{/each}
									</div>

									<div class="no-scrollbar mt-6 overflow-x-auto">
										<table class="min-w-full border-collapse text-left text-sm">
											<thead class="border-b border-white/8 bg-white/[0.03] text-white/42">
												<tr>
													<th class="px-4 py-3 font-bold">연도</th>
													<th class="px-4 py-3 font-bold">구성력</th>
													<th class="px-4 py-3 font-bold">명암/질감</th>
													<th class="px-4 py-3 font-bold">조형완성도</th>
													<th class="px-4 py-3 font-bold">주제해석</th>
												</tr>
											</thead>
											<tbody>
												{#each selectedUniversity.historyRows as row}
													<tr class={`border-b border-white/8 ${row.isMine ? 'bg-fuchsia-500/10 text-white' : 'text-white/72'}`}>
														<td class={`px-4 py-3 ${row.isMine ? 'font-black' : 'font-semibold'}`}>{row.year}</td>
														{#each row.scores as score, scoreIndex}
															<td class="px-4 py-3" style="font-variant-numeric: tabular-nums;">
																<span class={`${row.isMine ? 'font-black' : 'font-semibold'}`}>{score}</span>
																{#if !row.isMine}
																	<span class={`ml-2 text-xs font-bold ${row.diffs[scoreIndex] >= 0 ? 'text-emerald-300' : 'text-rose-300'}`}>
																		{row.diffs[scoreIndex] > 0 ? `+${row.diffs[scoreIndex]}` : row.diffs[scoreIndex]}
																	</span>
																{/if}
															</td>
														{/each}
													</tr>
												{/each}
											</tbody>
										</table>
									</div>
								</div>

								<div>
									<h3 class="font-display text-2xl font-bold tracking-[-0.04em] text-white">맞춤 개선 가이드</h3>
									<div class="mt-4 flex flex-wrap gap-2">
										{#each expertGuideTabs as tab}
											<button
												type="button"
												class={`rounded-2xl border px-4 py-2.5 text-sm font-bold transition-colors duration-200 ${
													expertTab === tab.id
														? 'border-fuchsia-400/40 bg-fuchsia-500/12 text-fuchsia-100'
														: 'border-white/8 bg-white/[0.03] text-white/52 hover:bg-white/8 hover:text-white'
												}`}
												aria-pressed={expertTab === tab.id}
												onclick={() => {
													selectExpertGuide(tab.id);
												}}
											>
												<span class="inline-flex items-center gap-2">
													<tab.icon class="size-4" aria-hidden="true" />
													{tab.label}
												</span>
											</button>
										{/each}
									</div>

									<div class="mt-6 grid gap-4">
										{#each selectedUniversity.expertGuides[expertTab] as guide}
											<div
												class={`rounded-r-[22px] border-l-4 bg-white/[0.03] p-5 ${guide.needImprovement ? 'border-orange-400' : 'border-emerald-400'}`}
											>
												<div class="flex flex-wrap items-center justify-between gap-3">
													<span class="text-lg font-bold text-white">{guide.axis}</span>
													<span class={`rounded-full px-3 py-1 text-xs font-black ${guide.needImprovement ? 'bg-orange-500/16 text-orange-200' : 'bg-emerald-500/14 text-emerald-200'}`}>
														{guide.status}
													</span>
												</div>
												<p class="soft-text mt-3">{guide.text}</p>
											</div>
										{/each}
									</div>
								</div>
							</div>
						</GlassCard>

						{#if !proUnlocked}
							<div class="absolute inset-0 z-10 flex items-center justify-center">
								<div class="glass-panel rounded-[32px] px-7 py-6 text-center">
									<Sparkles class="mx-auto size-9 text-fuchsia-300" aria-hidden="true" />
									<h3 class="mt-4 font-display text-2xl font-black tracking-[-0.04em] text-white">
										Pro 분석으로 합격 전략 확인
									</h3>
									<p class="soft-text mt-2">
										전문가 의견, 역대 기조, 대학별 맞춤 인사이트를 열 수 있습니다.
									</p>
									<button
										type="button"
										class="mt-5 rounded-full bg-gradient-to-r from-fuchsia-500 to-azure-450 px-6 py-3 font-semibold text-white transition-transform duration-200 hover:scale-[1.02]"
										onclick={() => {
											updateQuery({ tier: 'PRO' });
										}}
									>
										대학별 심층 분석 — +$40
									</button>
								</div>
							</div>
						{/if}
					</div>
				</section>

				<div class="flex justify-center">
					<button
						type="button"
						class="rounded-full border border-white/12 bg-white/6 px-8 py-3 font-semibold text-white transition-colors duration-200 hover:bg-white/10"
						onclick={resetDiagnosis}
					>
						새 작품 진단하기
					</button>
				</div>
			</div>
		{/if}
	</div>
</section>
