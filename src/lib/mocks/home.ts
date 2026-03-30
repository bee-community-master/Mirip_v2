import type { HomeFeature } from '$lib/types';

export const homeHeroImages = [
	'https://images.unsplash.com/photo-1657584905470-ac4ef76ee2b4?q=80&w=1080&auto=format&fit=crop',
	'https://images.unsplash.com/photo-1593472807861-5bb884af28f6?q=80&w=1080&auto=format&fit=crop',
	'https://images.unsplash.com/photo-1666559822683-31578dec8d39?q=80&w=1080&auto=format&fit=crop'
];

export const homeFeatures: HomeFeature[] = [
	{
		icon: 'trend',
		title: '정확도 95%의 평가모델',
		description: 'DINOv2 ViT-L과 5만 장의 합격작 데이터를 바탕으로 객관적인 평가 기준을 만듭니다.'
	},
	{
		icon: 'sparkles',
		title: '4축 스킬 분석',
		description: '구성력, 명암/질감, 조형완성도, 주제해석력을 S, A, B, C 등급 기준으로 해석합니다.'
	},
	{
		icon: 'trophy',
		title: '실시간 공모전 매칭',
		description: '작품 톤과 역량에 맞는 공모전을 묶어서 보여주고 포트폴리오 흐름까지 연결합니다.'
	}
];
