<script lang="ts">
	import type { DiagnosisRadarPoint } from '$lib/diagnosis/types';
	import { getRadarGeometry } from '$lib/utils/radar';

	let {
		data,
		size = 320
	}: {
		data: DiagnosisRadarPoint[];
		size?: number;
	} = $props();

	const geometry = $derived(getRadarGeometry(data, { size }));
</script>

<div class="relative h-full w-full">
	<svg viewBox={`0 0 ${geometry.size} ${geometry.size}`} class="h-full w-full overflow-visible">
		{#each geometry.rings as ring}
			<polygon points={ring} class="fill-transparent stroke-white/8" stroke-width="1" />
		{/each}

		{#each geometry.axes as axis}
			<line
				x1={axis.x1}
				y1={axis.y1}
				x2={axis.x2}
				y2={axis.y2}
				class="stroke-white/10"
				stroke-width="1"
			/>
		{/each}

		<polygon
			points={geometry.polygon}
			fill="rgba(211,77,243,0.22)"
			stroke="rgba(211,77,243,0.95)"
			stroke-width="2"
		/>

		{#each geometry.points as point}
			<circle cx={point.x} cy={point.y} r="5" fill="rgba(211,77,243,1)" />
			<g>
				<text
					x={point.labelX}
					y={point.labelY}
					text-anchor={point.textAnchor}
					class="fill-white/72 text-[12px] font-semibold"
				>
					{point.subject}
				</text>
				<text
					x={point.labelX}
					y={point.scoreY}
					text-anchor={point.textAnchor}
					class="fill-fuchsia-300 text-[12px] font-bold"
				>
					{point.score}
				</text>
			</g>
		{/each}
	</svg>
</div>
