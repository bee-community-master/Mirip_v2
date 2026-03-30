import type { CompetitionFilter, CompetitionItem } from '$lib/types';

export function filterCompetitions(items: CompetitionItem[], filter: CompetitionFilter) {
	if (filter === '전체') {
		return items;
	}

	return items.filter((item) => item.category === filter);
}
