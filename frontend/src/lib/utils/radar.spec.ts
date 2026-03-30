import { describe, expect, it } from 'vitest';
import { diagnosisMock } from '$lib/mocks/diagnosis';
import { getRadarGeometry } from '$lib/utils/radar';

describe('getRadarGeometry', () => {
	it('builds one point and one axis for each radar datum', () => {
		const geometry = getRadarGeometry(diagnosisMock.radar);

		expect(geometry.points).toHaveLength(diagnosisMock.radar.length);
		expect(geometry.axes).toHaveLength(diagnosisMock.radar.length);
		expect(geometry.rings).toHaveLength(4);
	});

	it('keeps plotted points inside the viewport bounds', () => {
		const geometry = getRadarGeometry(diagnosisMock.radar, { size: 320 });

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
