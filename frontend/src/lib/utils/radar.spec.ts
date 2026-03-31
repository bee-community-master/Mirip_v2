import { describe, expect, it } from 'vitest';
import { getRadarGeometry } from '$lib/utils/radar';

const radarFixture = [
	{ subject: '구성력', score: 79, fullMark: 100 },
	{ subject: '표현력', score: 74, fullMark: 100 },
	{ subject: '창의성', score: 73, fullMark: 100 },
	{ subject: '완성도', score: 83, fullMark: 100 }
];

describe('getRadarGeometry', () => {
	it('builds one point and one axis for each radar datum', () => {
		const geometry = getRadarGeometry(radarFixture);

		expect(geometry.points).toHaveLength(radarFixture.length);
		expect(geometry.axes).toHaveLength(radarFixture.length);
		expect(geometry.rings).toHaveLength(4);
	});

	it('keeps plotted points inside the viewport bounds', () => {
		const geometry = getRadarGeometry(radarFixture, { size: 320 });

		expect(
			geometry.points.every(
				(point) =>
					point.x >= 0 &&
					point.x <= geometry.size &&
					point.y >= 0 &&
					point.y <= geometry.size
			)
		).toBe(true);
	});
});
