<script lang="ts">
	import { onDestroy } from 'svelte';
	import {
		AlertCircle,
		CheckCircle2,
		LoaderCircle,
		RefreshCcw,
		Sparkles,
		TrendingUp,
		Upload
	} from 'lucide-svelte';

	import {
		completeUpload,
		createDiagnosisJob,
		createUploadSession,
		getDiagnosisJobStatus,
		isFakeUploadMode,
		uploadToSignedUrl
	} from '$lib/api/diagnosis';
	import type { DiagnosisJobDto } from '$lib/api/types';
	import GlassCard from '$lib/components/GlassCard.svelte';
	import RadarChart from '$lib/components/RadarChart.svelte';
	import SectionHeading from '$lib/components/SectionHeading.svelte';
	import { mapDiagnosisResult } from '$lib/diagnosis/adapter';
	import { transitionDiagnosisStage, type DiagnosisStage } from '$lib/diagnosis/flow';
	import {
		buildDefaultDiagnosisJobRequest,
		diagnosisPollIntervalInMs,
		getDiagnosisFailureMessage,
		getDiagnosisJobStatusLabel,
		validateDiagnosisFile
	} from '$lib/diagnosis/logic';
	import type { DiagnosisResultView } from '$lib/diagnosis/types';

	let stage = $state<DiagnosisStage>('upload');
	let selectedFile = $state<File | null>(null);
	let previewUrl = $state<string | null>(null);
	let diagnosisResult = $state<DiagnosisResultView | null>(null);
	let currentJob = $state<DiagnosisJobDto | null>(null);
	let error = $state('');
	let isDragging = $state(false);
	let statusMessage = $state('');
	let pollCount = $state(0);
	let pollTimer: ReturnType<typeof setTimeout> | null = null;
	let activeRunId = 0;
	let isDestroyed = false;
	const diagnosisUpdatedAtFormatter = new Intl.DateTimeFormat('ko-KR', {
		dateStyle: 'medium',
		timeStyle: 'short'
	});

	const currentTier = $derived(diagnosisResult?.tier ?? null);
	const topProbability = $derived(diagnosisResult?.probabilities[0] ?? null);
	const formattedUpdatedAt = $derived(
		currentJob ? diagnosisUpdatedAtFormatter.format(new Date(currentJob.updated_at)) : null
	);
	const workerHintVisible = $derived(stage === 'analyzing' && pollCount >= 3);

	function clearPreview() {
		if (previewUrl) {
			URL.revokeObjectURL(previewUrl);
			previewUrl = null;
		}
	}

	function clearPollingTimer() {
		if (pollTimer) {
			clearTimeout(pollTimer);
			pollTimer = null;
		}
	}

	function invalidateCurrentRun() {
		activeRunId += 1;
		clearPollingTimer();
	}

	function resetDiagnosisState() {
		stage = 'upload';
		selectedFile = null;
		diagnosisResult = null;
		currentJob = null;
		error = '';
		statusMessage = '';
		pollCount = 0;
		isDragging = false;
	}

	function isRunActive(runId: number) {
		return !isDestroyed && runId === activeRunId;
	}

	function resetDiagnosis() {
		invalidateCurrentRun();
		clearPreview();
		resetDiagnosisState();
	}

	function extractErrorMessage(value: unknown) {
		if (value instanceof Error) {
			return value.message;
		}

		return '진단 요청 중 문제가 발생했습니다. 잠시 후 다시 시도해 주세요.';
	}

	async function pollDiagnosisJob(jobId: string, runId: number) {
		clearPollingTimer();

		try {
			const response = await getDiagnosisJobStatus(jobId);

			if (!isRunActive(runId)) {
				return;
			}

			currentJob = response.job;
			pollCount += 1;
			stage = transitionDiagnosisStage(stage, {
				type: 'job_polled',
				jobStatus: response.job.status,
				hasResult: response.result !== null
			});

			if (response.job.status === 'succeeded' && response.result) {
				diagnosisResult = mapDiagnosisResult(response.result);
				statusMessage = '';
				return;
			}

			if (response.job.status === 'failed' || response.job.status === 'expired') {
				error = getDiagnosisFailureMessage(response.job);
				statusMessage = '';
				return;
			}

			statusMessage = `${getDiagnosisJobStatusLabel(response.job.status)} 상태입니다. 결과가 준비되면 자동으로 갱신됩니다.`;
			pollTimer = setTimeout(() => {
				void pollDiagnosisJob(jobId, runId);
			}, diagnosisPollIntervalInMs);
		} catch (caughtError) {
			if (!isRunActive(runId)) {
				return;
			}

			stage = transitionDiagnosisStage(stage, { type: 'request_failed' });
			error = extractErrorMessage(caughtError);
			statusMessage = '';
		}
	}

	async function beginAnalysis(file: File) {
		try {
			validateDiagnosisFile(file);
		} catch (caughtError) {
			error = extractErrorMessage(caughtError);
			stage = 'upload';
			return;
		}

		let runId = 0;

		try {
			invalidateCurrentRun();
			clearPreview();
			resetDiagnosisState();
			runId = activeRunId;
			error = '';
			selectedFile = file;
			previewUrl = URL.createObjectURL(file);
			stage = transitionDiagnosisStage(stage, { type: 'upload_started' });
			statusMessage = '업로드 세션을 준비하고 있습니다.';

			const uploadResponse = await createUploadSession(file);

			if (!isRunActive(runId)) {
				return;
			}

			if (!isFakeUploadMode(uploadResponse.data.session, uploadResponse.headers)) {
				statusMessage = '이미지를 업로드하고 있습니다.';
				await uploadToSignedUrl(uploadResponse.data.session, file);
			}

			if (!isRunActive(runId)) {
				return;
			}

			statusMessage = '업로드를 확정하고 진단 작업을 생성하고 있습니다.';
			const completedUpload = await completeUpload(uploadResponse.data.upload.id);

			if (!isRunActive(runId)) {
				return;
			}

			const job = await createDiagnosisJob(buildDefaultDiagnosisJobRequest(completedUpload.upload.id));

			if (!isRunActive(runId)) {
				return;
			}

			currentJob = job;
			pollCount = 0;
			statusMessage = '진단 작업이 접수되었습니다. 결과를 확인하는 중입니다.';
			await pollDiagnosisJob(job.id, runId);
		} catch (caughtError) {
			if (!isRunActive(runId)) {
				return;
			}

			stage = transitionDiagnosisStage(stage, { type: 'request_failed' });
			error = extractErrorMessage(caughtError);
			statusMessage = '';
		}
	}

	function handleInputChange(event: Event) {
		const target = event.currentTarget as HTMLInputElement;
		const file = target.files?.[0];

		if (file) {
			void beginAnalysis(file);
		}

		target.value = '';
	}

	function handleDrop(event: DragEvent) {
		event.preventDefault();
		isDragging = false;

		const file = event.dataTransfer?.files?.[0];

		if (file) {
			void beginAnalysis(file);
		}
	}

	onDestroy(() => {
		isDestroyed = true;
		invalidateCurrentRun();
		clearPreview();
	});
</script>

<svelte:head>
	<title>Diagnosis | MIRIP v2</title>
	<meta
		name="description"
		content="업로드한 작품을 실제 Mirip v2 backend diagnosis API에 연결해 결과를 확인하는 화면입니다."
	/>
</svelte:head>

<section class="section-frame py-16 sm:py-20">
	<div class="mx-auto flex max-w-6xl flex-col gap-10">
		<SectionHeading
			badge="AI Diagnosis"
			title="작품을 업로드하고 현재 진단 결과를 확인하세요"
			subtitle="frontend가 backend upload, diagnosis job, worker polling 흐름에 직접 연결되어 있습니다."
			center={true}
			level="h1"
		/>

		{#if error}
			<div
				class="mx-auto flex w-full max-w-3xl items-start gap-3 rounded-[24px] border border-rose-400/20 bg-rose-500/10 px-5 py-4 text-sm font-medium text-rose-100"
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
				<p class="soft-text mt-3 max-w-md">PNG, JPG, JPEG, WebP, 최대 10 MB까지 지원합니다.</p>
			</label>
		{:else if stage === 'analyzing'}
			<div class="grid gap-6 lg:grid-cols-[1.05fr_0.95fr]">
				<GlassCard className="rounded-[30px] p-8">
					<div class="flex flex-col items-center justify-center py-8 text-center">
						<div class="relative mb-8 flex size-28 items-center justify-center rounded-full border border-white/10 bg-night-900/70">
							<LoaderCircle
								class="size-12 animate-spin text-fuchsia-300"
								aria-hidden="true"
							/>
						</div>
						<h2 class="font-display text-3xl font-bold tracking-[-0.05em] text-white">
							DINOv2 AI 분석 중
						</h2>
						<p class="soft-text mt-3 max-w-xl">
							{statusMessage || '업로드와 진단 요청을 처리하고 있습니다.'}
						</p>

						{#if currentJob}
							<div class="mt-8 grid w-full gap-3 text-left sm:grid-cols-2">
								<div class="rounded-[22px] border border-white/8 bg-white/[0.03] p-4">
									<div class="text-xs font-bold uppercase tracking-[0.24em] text-white/35">Job ID</div>
									<div class="mt-2 break-all text-sm font-medium text-white/78">{currentJob.id}</div>
								</div>
								<div class="rounded-[22px] border border-white/8 bg-white/[0.03] p-4">
									<div class="text-xs font-bold uppercase tracking-[0.24em] text-white/35">Status</div>
									<div class="mt-2 text-sm font-medium text-white/78">
										{getDiagnosisJobStatusLabel(currentJob.status)}
									</div>
								</div>
							</div>
						{/if}

						{#if workerHintVisible}
							<div class="mt-8 w-full rounded-[24px] border border-amber-400/20 bg-amber-500/10 p-4 text-left text-sm text-amber-100">
								<div class="font-semibold">로컬 worker를 아직 실행하지 않았다면 현재 상태가 계속 유지됩니다.</div>
								<p class="mt-2 text-amber-100/80">
									backend에서 `uv run python -m mirip_backend.worker.main` 또는 worker 실행 스크립트를
									별도 프로세스로 띄워 주세요.
								</p>
							</div>
						{/if}
					</div>
				</GlassCard>

				<GlassCard className="rounded-[30px] p-7 sm:p-8">
					<div class="mb-4 flex items-center justify-between">
						<h3 class="font-display text-2xl font-bold tracking-[-0.04em] text-white">Uploaded</h3>
						<span class="rounded-full bg-fuchsia-500/16 px-3 py-1 text-xs font-bold text-fuchsia-200">
							Processing
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

					{#if selectedFile}
						<div class="mt-4 rounded-[22px] border border-white/8 bg-white/[0.03] p-4 text-sm text-white/72">
							<div class="font-semibold text-white">{selectedFile.name}</div>
							<div class="mt-1 text-white/52">
								{Math.round(selectedFile.size / 1024)} KB · {selectedFile.type}
							</div>
						</div>
					{/if}
				</GlassCard>
			</div>
		{:else if stage === 'result' && diagnosisResult}
			<div class="flex flex-col gap-10">
				<div class="grid gap-6 lg:grid-cols-[1.08fr_0.92fr]">
					<GlassCard className="rounded-[30px] p-7 sm:p-8">
						<div class="grid gap-8 xl:grid-cols-[1fr_280px] xl:items-center">
							<div>
								<div class="h-[320px]">
									<RadarChart data={diagnosisResult.radarPoints} />
								</div>
							</div>

							<div class="space-y-4 rounded-[28px] border border-white/8 bg-night-900/70 p-5">
								<div>
									<div class="text-xs font-bold uppercase tracking-[0.24em] text-white/35">
										Predicted Tier
									</div>
									<div class="mt-2 font-display text-5xl font-black tracking-[-0.08em] text-white">
										{currentTier}
									</div>
								</div>

								{#if topProbability}
									<div class="rounded-[22px] border border-white/8 bg-white/[0.03] p-4">
										<div class="text-xs font-bold uppercase tracking-[0.24em] text-white/35">
											Top Match
										</div>
										<div class="mt-2 text-lg font-semibold text-white">
											{topProbability.university}
										</div>
										<p class="mt-1 text-sm text-white/55">{topProbability.department}</p>
										<div class="mt-3 inline-flex rounded-full bg-fuchsia-500/14 px-3 py-1 text-sm font-bold text-fuchsia-200">
											{topProbability.percentLabel}
										</div>
									</div>
								{/if}

								<div class="rounded-[22px] border border-white/8 bg-white/[0.03] p-4">
									<div class="text-xs font-bold uppercase tracking-[0.24em] text-white/35">
										Last Updated
									</div>
									<div class="mt-2 text-sm font-medium text-white/72">
										{formattedUpdatedAt ?? '-'}
									</div>
								</div>

								<div class="rounded-[22px] border border-white/8 bg-white/[0.03] p-4">
									<div class="mb-3 text-xs font-bold uppercase tracking-[0.24em] text-white/35">
										Uploaded Preview
									</div>
									<div class="overflow-hidden rounded-[18px] border border-white/8">
										<img
											src={previewUrl ?? 'https://images.unsplash.com/photo-1593472807861-5bb884af28f6?q=80&w=800&auto=format&fit=crop'}
											alt="진단에 사용한 업로드 이미지"
											width="800"
											height="1000"
											class="aspect-[4/5] w-full object-cover"
										/>
									</div>
									{#if selectedFile}
										<div class="mt-3 text-xs text-white/52">
											{selectedFile.name} · {Math.round(selectedFile.size / 1024)} KB
										</div>
									{/if}
								</div>
							</div>
						</div>
					</GlassCard>

					<GlassCard className="rounded-[30px] p-7 sm:p-8">
						<div class="flex items-center gap-3">
							<div class="flex size-12 items-center justify-center rounded-2xl bg-fuchsia-500/12 text-fuchsia-200">
								<Sparkles class="size-6" aria-hidden="true" />
							</div>
							<div>
								<h3 class="font-display text-2xl font-bold tracking-[-0.04em] text-white">Diagnosis Summary</h3>
								<p class="text-sm text-white/45">backend worker가 생성한 실제 결과 요약입니다.</p>
							</div>
						</div>

						<div class="mt-6 space-y-4">
							<div class="rounded-[24px] border border-white/8 bg-white/[0.03] p-5">
								<div class="text-xs font-bold uppercase tracking-[0.24em] text-white/35">Summary</div>
								<p class="soft-text mt-3">
									{diagnosisResult.summary ?? '요약 문구가 아직 제공되지 않았습니다.'}
								</p>
							</div>

							<div class="rounded-[24px] border border-white/8 bg-white/[0.03] p-5">
								<div class="text-xs font-bold uppercase tracking-[0.24em] text-white/35">Overall Feedback</div>
								<p class="soft-text mt-3">
									{diagnosisResult.feedback.overall ?? '세부 피드백 없이 기본 점수와 확률 결과만 제공되었습니다.'}
								</p>
							</div>

							<button
								type="button"
								class="inline-flex w-full items-center justify-center gap-2 rounded-full bg-white px-5 py-3 font-semibold text-black transition-transform duration-200 hover:scale-[1.02]"
								onclick={resetDiagnosis}
							>
								<RefreshCcw class="size-4" aria-hidden="true" />
								새 작품 다시 진단하기
							</button>
						</div>
					</GlassCard>
				</div>

				<div class="grid gap-6 lg:grid-cols-[1.05fr_0.95fr]">
					<GlassCard className="rounded-[30px] p-7 sm:p-8">
						<h3 class="inline-flex items-center gap-2 font-display text-2xl font-bold tracking-[-0.04em] text-white">
							<TrendingUp class="size-5 text-fuchsia-300" aria-hidden="true" />
							대학교/학과별 확률
						</h3>

						<div class="mt-6 space-y-4">
							{#if diagnosisResult.probabilities.length}
								{#each diagnosisResult.probabilities as probability}
									<div class="rounded-[24px] border border-white/8 bg-white/[0.03] p-5">
										<div class="flex flex-wrap items-start justify-between gap-4">
											<div>
												<h4 class="text-lg font-semibold text-white">{probability.university}</h4>
												<p class="mt-1 text-sm text-white/52">{probability.department}</p>
											</div>
											<div class="rounded-full bg-fuchsia-500/14 px-3 py-1 text-sm font-bold text-fuchsia-200">
												{probability.percentLabel}
											</div>
										</div>
										<div class="mt-4 h-3 overflow-hidden rounded-full bg-white/8">
											<div
												class="h-full rounded-full bg-gradient-to-r from-fuchsia-400 to-azure-450"
												style={`width: ${probability.percentLabel};`}
											></div>
										</div>
									</div>
								{/each}
							{:else}
								<div class="rounded-[24px] border border-white/8 bg-white/[0.03] p-5 text-sm text-white/60">
									backend가 확률 항목을 반환하지 않았습니다.
								</div>
							{/if}
						</div>
					</GlassCard>

					<div class="space-y-6">
						<GlassCard className="rounded-[30px] p-7 sm:p-8">
							<div class="mb-4 flex items-center gap-2 text-lg font-black text-blue-200">
								<CheckCircle2 class="size-5" aria-hidden="true" />
								강점
							</div>
							{#if diagnosisResult.feedback.strengths.length}
								<ul class="space-y-3">
									{#each diagnosisResult.feedback.strengths as item}
										<li class="soft-text flex gap-3">
											<span class="mt-2 size-1.5 rounded-full bg-blue-200" aria-hidden="true"></span>
											<span>{item}</span>
										</li>
									{/each}
								</ul>
							{:else}
								<p class="soft-text">현재 결과에는 강점 항목이 포함되지 않았습니다.</p>
							{/if}
						</GlassCard>

						<GlassCard className="rounded-[30px] p-7 sm:p-8">
							<div class="mb-4 flex items-center gap-2 text-lg font-black text-orange-200">
								<AlertCircle class="size-5" aria-hidden="true" />
								보완점
							</div>
							{#if diagnosisResult.feedback.improvements.length}
								<ul class="space-y-3">
									{#each diagnosisResult.feedback.improvements as item}
										<li class="soft-text flex gap-3">
											<span class="mt-2 size-1.5 rounded-full bg-orange-200" aria-hidden="true"></span>
											<span>{item}</span>
										</li>
									{/each}
								</ul>
							{:else}
								<p class="soft-text">현재 결과에는 보완점 항목이 포함되지 않았습니다.</p>
							{/if}
						</GlassCard>
					</div>
				</div>
			</div>
		{:else}
			<div class="mx-auto flex max-w-2xl flex-col items-center gap-4 rounded-[30px] border border-white/8 bg-white/[0.03] p-8 text-center">
				<AlertCircle class="size-10 text-rose-200" aria-hidden="true" />
				<h2 class="font-display text-3xl font-bold tracking-[-0.04em] text-white">진단을 완료하지 못했습니다</h2>
				<p class="soft-text">
					요청이 중단되었거나 worker 실행이 지연되고 있습니다. 잠시 후 다시 시도하거나 worker 상태를
					확인해 주세요.
				</p>
				<button
					type="button"
					class="inline-flex items-center justify-center gap-2 rounded-full bg-white px-5 py-3 font-semibold text-black transition-transform duration-200 hover:scale-[1.02]"
					onclick={resetDiagnosis}
				>
					<RefreshCcw class="size-4" aria-hidden="true" />
					처음부터 다시 시작
				</button>
			</div>
		{/if}
	</div>
</section>
