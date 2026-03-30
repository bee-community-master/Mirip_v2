<script lang="ts">
	import { Flame, Sparkles, UserRound } from 'lucide-svelte';
	import Modal from '$lib/components/Modal.svelte';
	import { mockUserSession } from '$lib/mocks/user';

	let {
		open,
		onClose
	}: {
		open: boolean;
		onClose: () => void;
	} = $props();

	const benefits = [
		'공모전과 포트폴리오를 한 번에 연결',
		'AI 진단 결과를 저장하고 비교',
		'내 작업 흐름을 프로필 형태로 정리'
	];
</script>

<Modal {open} title="MIRIP Mock Sign In" {onClose}>
	<div class="grid gap-6 lg:grid-cols-[1.1fr_0.9fr]">
		<div class="space-y-5">
			<p class="soft-text">
				실제 인증 대신 로컬 상태만 사용하는 데모 모달입니다. 클릭 경고 없이 진입 흐름이
				완결되도록 mock 세션 정보를 보여줍니다.
			</p>

			<div class="space-y-3">
				{#each benefits as benefit}
					<div class="flex items-center gap-3 rounded-2xl border border-white/8 bg-white/[0.03] px-4 py-3">
						<Sparkles class="size-4 text-fuchsia-300" />
						<span class="text-sm font-medium text-white/78">{benefit}</span>
					</div>
				{/each}
			</div>

			<div class="grid gap-3 sm:grid-cols-2">
				<button
					type="button"
					class="rounded-full bg-white px-5 py-3 font-semibold text-black transition hover:scale-[1.02]"
					onclick={onClose}
				>
					Google로 계속
				</button>
				<button
					type="button"
					class="rounded-full border border-white/12 bg-white/5 px-5 py-3 font-semibold text-white transition hover:bg-white/10"
					onclick={onClose}
				>
					게스트로 보기
				</button>
			</div>
		</div>

		<div class="rounded-[28px] border border-white/10 bg-white/[0.035] p-5">
			<div class="mb-4 flex items-center gap-4">
				<img
					class="size-16 rounded-full border border-fuchsia-400/40 object-cover"
					src={mockUserSession.avatar}
					alt={mockUserSession.displayName}
				/>
				<div>
					<div class="font-display text-2xl font-bold tracking-[-0.04em] text-white">
						{mockUserSession.displayName}
					</div>
					<p class="text-sm font-medium text-white/55">{mockUserSession.status}</p>
				</div>
			</div>

			<div class="space-y-3 rounded-[24px] border border-white/8 bg-night-900/70 p-4">
				<div class="flex items-center justify-between text-sm text-white/60">
					<span class="inline-flex items-center gap-2">
						<UserRound class="size-4 text-white/40" />
						Profile Tier
					</span>
					<span class="font-semibold text-white">{mockUserSession.tierLabel}</span>
				</div>
				<div class="flex items-center justify-between text-sm text-white/60">
					<span class="inline-flex items-center gap-2">
						<Flame class="size-4 text-orange-300" />
						Current Streak
					</span>
					<span class="font-semibold text-white">{mockUserSession.streak} days</span>
				</div>
			</div>
		</div>
	</div>
</Modal>
