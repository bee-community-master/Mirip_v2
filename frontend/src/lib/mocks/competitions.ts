import type { CompetitionFilter, CompetitionItem } from '$lib/types';

export const competitionFilters: CompetitionFilter[] = ['전체', '디지털', '회화', '캐릭터'];

export const competitionItems: CompetitionItem[] = [
	{
		id: 'digital-youth-2026',
		title: '2026 전국 청소년 디지털 아트 공모전',
		org: '한국디지털아트협회',
		dday: 'D-15',
		category: '디지털',
		tags: ['디지털드로잉', '일러스트'],
		image:
			'https://images.unsplash.com/photo-1774021792660-52381ac54949?q=80&w=800&auto=format&fit=crop',
		prize: '총 상금 500만원',
		summary: '탄탄한 완성도와 화면 밀도가 강한 작품에 유리한 전국 단위 디지털 공모전입니다.',
		deadline: '2026.04.14',
		region: '서울',
		participants: '1,240명 예정',
		difficulty: '중상',
		matchingReason: [
			'조형 완성도가 높은 작품에 가산점이 큽니다.',
			'네온 계열 색감과 도시적 분위기가 최근 수상작과 잘 맞습니다.',
			'포트폴리오 첫 장으로 쓰기 좋은 대표작을 만들기 좋습니다.'
		]
	},
	{
		id: 'future-art-15',
		title: '제 15회 미래 미술인 발굴 프로젝트',
		org: '현대미술관',
		dday: 'D-3',
		category: '회화',
		tags: ['순수미술', '회화'],
		image:
			'https://images.unsplash.com/photo-1525434486320-567654c1f256?q=80&w=800&auto=format&fit=crop',
		prize: '입학 특전',
		summary: '과정 설명과 화면의 주제 해석이 중요한 회화 중심 프로젝트입니다.',
		deadline: '2026.04.02',
		region: '과천',
		participants: '680명 예정',
		difficulty: '상',
		matchingReason: [
			'주제 해석력 점수가 높은 학생에게 적합합니다.',
			'설정 노트와 발상 과정 설명이 함께 평가됩니다.',
			'입시 포트폴리오에서 사고의 확장을 보여주기 좋습니다.'
		]
	},
	{
		id: 'neon-character-contest',
		title: '네온 펑크 캐릭터 디자인 콘테스트',
		org: '게임개발자연대',
		dday: 'D-28',
		category: '캐릭터',
		tags: ['캐릭터', '원화'],
		image:
			'https://images.unsplash.com/photo-1666559822683-31578dec8d39?q=80&w=800&auto=format&fit=crop',
		prize: '총 상금 1,000만원',
		summary: '강한 콘셉트와 감정선이 살아있는 캐릭터 설정을 요구하는 상업 원화형 공모전입니다.',
		deadline: '2026.04.27',
		region: '성수',
		participants: '920명 예정',
		difficulty: '중',
		matchingReason: [
			'채색 밀도와 분위기 연출에 강점이 있으면 유리합니다.',
			'세계관 설정 문서까지 함께 제출할 수 있어 포트폴리오 전개가 좋습니다.',
			'브랜딩형 작업보다 캐릭터 중심 프로젝트를 원하는 경우 적합합니다.'
		]
	},
	{
		id: 'visual-lab-open-call',
		title: 'Visual Lab 오픈 콜 포스터 챌린지',
		org: 'Visual Lab Seoul',
		dday: 'D-9',
		category: '디지털',
		tags: ['포스터', '그래픽'],
		image:
			'https://images.unsplash.com/photo-1498050108023-c5249f4df085?q=80&w=800&auto=format&fit=crop',
		prize: '전시 및 굿즈 제작',
		summary: '그래픽 리듬과 타이포 결합을 강하게 보는 포스터 챌린지입니다.',
		deadline: '2026.04.08',
		region: '온라인',
		participants: '450명 예정',
		difficulty: '중',
		matchingReason: [
			'시각디자인 계열 지원자에게 바로 연결되는 문법을 요구합니다.',
			'한 장의 포스터에 서사를 압축하는 능력을 보여주기 좋습니다.',
			'짧은 마감 주기로 집중 훈련하기에 적합합니다.'
		]
	}
];
