import type { PortfolioProfile, PortfolioWork } from '$lib/types';

export const portfolioProfile: PortfolioProfile = {
	name: 'Chanspick',
	avatar:
		'https://images.unsplash.com/photo-1593472807861-5bb884af28f6?q=80&w=320&auto=format&fit=crop',
	role: '시각디자인 지망생 / AI 진단 Tier A',
	stats: [
		{ label: 'Works', value: '42' },
		{ label: 'Likes', value: '12.4k' },
		{ label: 'Awards', value: '3' }
	]
};

export const portfolioWorks: PortfolioWork[] = [
	{
		id: 'cyber-city',
		title: 'Cyber City',
		image:
			'https://images.unsplash.com/photo-1657584905470-ac4ef76ee2b4?q=80&w=800&auto=format&fit=crop',
		likes: 124,
		comments: 28,
		height: 'tall',
		year: '2026',
		tags: ['도시', '컨셉아트', '네온'],
		description: '수직 도시의 야간 리듬을 연구한 대표작입니다. 라이트 흐름과 거리 원근을 강조했습니다.'
	},
	{
		id: 'study-04',
		title: 'Study #04',
		image:
			'https://images.unsplash.com/photo-1593472807861-5bb884af28f6?q=80&w=800&auto=format&fit=crop',
		likes: 89,
		comments: 17,
		height: 'medium',
		year: '2025',
		tags: ['관찰드로잉', '질감'],
		description: '정물 관찰을 기반으로 질감과 밝기 차이를 축적한 연습 시리즈입니다.'
	},
	{
		id: 'abstract-emotion',
		title: 'Abstract Emotion',
		image:
			'https://images.unsplash.com/photo-1525434486320-567654c1f256?q=80&w=800&auto=format&fit=crop',
		likes: 342,
		comments: 51,
		height: 'tall',
		year: '2026',
		tags: ['추상', '감정', '회화'],
		description: '감정의 이동을 색면과 방향성으로 번역한 작업입니다. 주제 해석력 포트폴리오에 쓰고 있습니다.'
	},
	{
		id: 'neon-dreams',
		title: 'Neon Dreams',
		image:
			'https://images.unsplash.com/photo-1666559822683-31578dec8d39?q=80&w=800&auto=format&fit=crop',
		likes: 256,
		comments: 42,
		height: 'medium',
		year: '2026',
		tags: ['캐릭터', '무드', '컬러'],
		description: '네온 사인과 인물 실루엣을 결합한 무드 보드형 일러스트레이션입니다.'
	},
	{
		id: 'museum-visit',
		title: 'Museum Visit',
		image:
			'https://images.unsplash.com/photo-1774021792660-52381ac54949?q=80&w=800&auto=format&fit=crop',
		likes: 112,
		comments: 19,
		height: 'medium',
		year: '2025',
		tags: ['공간', '전시', '기록'],
		description: '전시 공간의 동선을 관찰하고, 시선 이동을 포스터 구성 실험으로 이어간 작업입니다.'
	},
	{
		id: 'quick-sketch',
		title: 'Quick Sketch',
		image:
			'https://images.unsplash.com/photo-1591776578494-f824f2734ef4?q=80&w=800&auto=format&fit=crop',
		likes: 45,
		comments: 11,
		height: 'medium',
		year: '2024',
		tags: ['스케치', '드로잉', '손풀기'],
		description: '짧은 시간 안에 화면 중심을 잡는 감각을 유지하려고 매일 반복한 스케치 기록입니다.'
	}
];
