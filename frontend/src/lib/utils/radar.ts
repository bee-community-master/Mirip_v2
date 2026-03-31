import type { DiagnosisRadarPoint } from '$lib/diagnosis/types';

interface RadarGeometryOptions {
	size?: number;
	levels?: number;
	labelOffset?: number;
}

export interface RadarGeometryPoint extends DiagnosisRadarPoint {
	x: number;
	y: number;
	labelX: number;
	labelY: number;
	scoreY: number;
	textAnchor: 'start' | 'middle' | 'end';
}

export interface RadarGeometry {
	size: number;
	center: number;
	rings: string[];
	axes: Array<{ x1: number; y1: number; x2: number; y2: number }>;
	polygon: string;
	points: RadarGeometryPoint[];
}

function round(value: number) {
	return Number(value.toFixed(3));
}

function polarToCartesian(center: number, radius: number, angle: number) {
	return {
		x: round(center + radius * Math.cos(angle)),
		y: round(center + radius * Math.sin(angle))
	};
}

export function getRadarGeometry(
	data: DiagnosisRadarPoint[],
	{ size = 320, levels = 4, labelOffset = 28 }: RadarGeometryOptions = {}
): RadarGeometry {
	const center = size / 2;
	const outerRadius = size * 0.32;
	const baseAngles = data.map((_, index) => -Math.PI / 2 + (index * Math.PI * 2) / data.length);

	const axes = baseAngles.map((angle) => {
		const end = polarToCartesian(center, outerRadius, angle);
		return { x1: center, y1: center, x2: end.x, y2: end.y };
	});

	const rings = Array.from({ length: levels }, (_, levelIndex) => {
		const radius = (outerRadius * (levelIndex + 1)) / levels;
		return baseAngles
			.map((angle) => {
				const point = polarToCartesian(center, radius, angle);
				return `${point.x},${point.y}`;
			})
			.join(' ');
	});

	const points = data.map((entry, index) => {
		const angle = baseAngles[index];
		const plotted = polarToCartesian(center, (outerRadius * entry.score) / entry.fullMark, angle);
		const labelPoint = polarToCartesian(center, outerRadius + labelOffset, angle);
		const verticalBias = Math.abs(Math.cos(angle)) < 0.1;
		const textAnchor: RadarGeometryPoint['textAnchor'] =
			Math.abs(Math.cos(angle)) < 0.22 ? 'middle' : Math.cos(angle) > 0 ? 'start' : 'end';

		return {
			...entry,
			x: plotted.x,
			y: plotted.y,
			labelX: labelPoint.x,
			labelY: verticalBias ? labelPoint.y - 8 : labelPoint.y,
			scoreY: verticalBias ? labelPoint.y + 12 : labelPoint.y + 16,
			textAnchor
		};
	});

	return {
		size,
		center,
		rings,
		axes,
		polygon: points.map((point) => `${point.x},${point.y}`).join(' '),
		points
	};
}
