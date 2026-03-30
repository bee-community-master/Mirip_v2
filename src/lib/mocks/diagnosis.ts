import type { DiagnosisMockData } from '$lib/types';

export const diagnosisMock: DiagnosisMockData = {
	radar: [
		{ subject: '구성력', score: 79, fullMark: 100 },
		{ subject: '명암/질감', score: 74, fullMark: 100 },
		{ subject: '조형완성도', score: 73, fullMark: 100 },
		{ subject: '주제해석', score: 83, fullMark: 100 }
	],
	tierResult: {
		predictedGrade: 'A',
		probability: 55,
		confidence: 94,
		segments: [
			{ label: '', value: 5, fillClass: 'bg-yellow-400/15', textClass: 'text-white/30' },
			{ label: 'A 55%', value: 55, fillClass: 'bg-white text-black', textClass: 'text-black' },
			{
				label: 'B 25%',
				value: 25,
				fillClass: 'bg-orange-400/18',
				textClass: 'text-orange-100'
			},
			{ label: 'C 15%', value: 15, fillClass: 'bg-blue-500/18', textClass: 'text-blue-100' }
		]
	},
	standardMatches: [
		{
			key: 'HONGIK',
			name: '홍익대학교 시각디자인과',
			match: 87,
			rank: '28%',
			applicants: '1,240명',
			ratio: '8.2:1'
		},
		{
			key: 'KONKUK',
			name: '건국대학교 산업디자인과',
			match: 82,
			rank: '35%',
			applicants: '980명',
			ratio: '7.4:1'
		},
		{
			key: 'KOOKMIN',
			name: '국민대학교 시각디자인과',
			match: 78,
			rank: '42%',
			applicants: '1,180명',
			ratio: '9.1:1'
		}
	],
	universities: {
		HONGIK: {
			key: 'HONGIK',
			label: '홍익대',
			name: '홍익대학교 시각디자인과',
			match: 87,
			rank: '상위 28%',
			stats: [
				{ label: '지원자', value: '1,240명' },
				{ label: '합격률', value: '8.2%' },
				{ label: '예상등수', value: '상위 28%', accent: true }
			],
			axisDistances: [
				{ label: '구성력', value: '+8 필요', needImprovement: true },
				{ label: '명암/질감', value: '합격권', needImprovement: false },
				{ label: '조형완성도', value: '합격권', needImprovement: false },
				{ label: '주제해석', value: '+3 필요', needImprovement: true }
			],
			strengths: ['조형 완성도가 합격작 평균 이상', '명암 표현이 안정적이고 화면의 밀도가 좋음'],
			improvements: ['구성력이 커트라인 근처에 머물러 있음', '주제 해석에서 2차 연상 확장이 더 필요함'],
			historyNotes: [
				{
					label: '출제 경향 변화',
					tone: 'blue',
					text: '최근 3년간 추상적·감성적 주제 비중이 증가하며, 구상 위주에서 혼합형으로 빠르게 이동하고 있습니다.'
				},
				{
					label: '최근 변화',
					tone: 'gold',
					text: "2024 수시부터 '과정 중심 평가' 도입. 완성도보다 사고 과정의 독창성을 더 높게 평가하는 경향입니다."
				},
				{
					label: '핵심 인사이트',
					tone: 'rose',
					text: '주제해석 커트라인이 꾸준히 상승 중이라 올해는 가장 중요한 축이 될 가능성이 높습니다.'
				}
			],
			historyRows: [
				{ year: '나의 점수', scores: [79, 74, 73, 83], diffs: [0, 0, 0, 0], isMine: true },
				{ year: '2024 커트', scores: [78, 72, 80, 82], diffs: [1, 2, -7, 1] },
				{ year: '2023 커트', scores: [76, 70, 78, 75], diffs: [3, 4, -5, 8] },
				{ year: '2022 커트', scores: [80, 68, 82, 70], diffs: [-1, 6, -9, 13] }
			],
			expertGuides: {
				AI: [
					{
						axis: '구성력',
						status: '+8 필요',
						needImprovement: true,
						text: '삼각 구도 외에 사선 흐름을 넣어 화면 안에서 시선이 순환하도록 설계하면 5~8점 상승 여지가 있습니다.'
					},
					{
						axis: '명암/질감',
						status: '합격권',
						needImprovement: false,
						text: '현재 수준은 안정적입니다. 반사광과 투영 그림자의 색온도 차이를 더 벌리면 완성도가 올라갑니다.'
					},
					{
						axis: '조형완성도',
						status: '합격권',
						needImprovement: false,
						text: '전체 비례는 안정적입니다. 마감 10분을 별도로 확보하는 루틴을 만들면 흔들림이 줄어듭니다.'
					},
					{
						axis: '주제해석',
						status: '+3 필요',
						needImprovement: true,
						text: '주제어에서 직접 연상에 멈추지 말고 감정과 맥락까지 확장해야 최근 홍익대 경향에 더 잘 맞습니다.'
					}
				],
				INSTRUCTOR: [
					{
						axis: '구성력',
						status: '+8 필요',
						needImprovement: true,
						text: '정물 배치 초안만 5분 안에 3개 이상 잡는 훈련이 필요합니다. 가장 강한 시선 흐름을 먼저 고르세요.'
					},
					{
						axis: '명암/질감',
						status: '합격권',
						needImprovement: false,
						text: '질감 표현은 이미 좋습니다. 질감 간 명도 차를 조금 더 단순화하면 화면이 더 선명하게 읽힙니다.'
					},
					{
						axis: '조형완성도',
						status: '합격권',
						needImprovement: false,
						text: '형태를 망치지 않는 선에서 하이라이트를 더 과감하게 남기면 채점자 시선이 빨리 붙습니다.'
					},
					{
						axis: '주제해석',
						status: '+3 필요',
						needImprovement: true,
						text: '문제 지문에서 핵심 명사를 2개로 요약한 뒤, 그 두 명사가 한 장면에서 충돌하도록 설정해보세요.'
					}
				],
				PROFESSOR: [
					{
						axis: '구성력',
						status: '+8 필요',
						needImprovement: true,
						text: '형태는 잘 잡았지만 시선이 도착한 이후의 여운이 짧습니다. 화면 중심과 비중 분배를 더 밀도 있게 조절해야 합니다.'
					},
					{
						axis: '명암/질감',
						status: '합격권',
						needImprovement: false,
						text: '소재 차이는 분명합니다. 다만 빛의 방향을 더 명확하게 잡으면 조형 전체가 더 설득력 있게 정리됩니다.'
					},
					{
						axis: '조형완성도',
						status: '합격권',
						needImprovement: false,
						text: '불필요한 선이 적어 평가자 입장에서 안정적으로 보입니다. 마지막 단계의 과감함만 조금 더 필요합니다.'
					},
					{
						axis: '주제해석',
						status: '+3 필요',
						needImprovement: true,
						text: '문제의 의미를 읽는 능력은 보이지만, 홍익대는 최근 추상화된 해석을 선호합니다. 직접적인 상징을 한 단계 더 비틀어야 합니다.'
					}
				]
			}
		},
		KONKUK: {
			key: 'KONKUK',
			label: '건국대',
			name: '건국대학교 산업디자인과',
			match: 82,
			rank: '상위 35%',
			stats: [
				{ label: '지원자', value: '980명' },
				{ label: '합격률', value: '10.4%' },
				{ label: '예상등수', value: '상위 35%', accent: true }
			],
			axisDistances: [
				{ label: '구성력', value: '+5 필요', needImprovement: true },
				{ label: '명암/질감', value: '합격권', needImprovement: false },
				{ label: '조형완성도', value: '+2 필요', needImprovement: true },
				{ label: '주제해석', value: '합격권', needImprovement: false }
			],
			strengths: ['주제 해석이 산업디자인 계열 문제와 잘 맞음', '명암 단계가 명확해 모형성이 잘 드러남'],
			improvements: ['구성 초안 속도가 느리면 손해를 볼 수 있음', '형태 정리 단계에서 디테일 과잉이 보임'],
			historyNotes: [
				{
					label: '출제 경향 변화',
					tone: 'blue',
					text: '건국대는 최근 실용성보다 화면 문제 해결력과 프로세스 설명 비중을 조금씩 높이고 있습니다.'
				},
				{
					label: '최근 변화',
					tone: 'gold',
					text: '아이디어 스케치와 최종안의 연결성이 강할수록 높은 점수를 받는 패턴이 확인됩니다.'
				},
				{
					label: '핵심 인사이트',
					tone: 'rose',
					text: '빠른 초안 전개를 먼저 보여준 뒤, 최종안에서 완성도를 챙기는 전략이 유효합니다.'
				}
			],
			historyRows: [
				{ year: '나의 점수', scores: [79, 74, 73, 83], diffs: [0, 0, 0, 0], isMine: true },
				{ year: '2024 커트', scores: [74, 71, 75, 79], diffs: [5, 3, -2, 4] },
				{ year: '2023 커트', scores: [72, 69, 74, 76], diffs: [7, 5, -1, 7] },
				{ year: '2022 커트', scores: [77, 68, 72, 74], diffs: [2, 6, 1, 9] }
			],
			expertGuides: {
				AI: [
					{
						axis: '구성력',
						status: '+5 필요',
						needImprovement: true,
						text: '건국대는 문제 해결 과정을 빠르게 보여주는 구성이 중요합니다. 아이디어 분기점을 더 선명하게 드러내세요.'
					},
					{
						axis: '명암/질감',
						status: '합격권',
						needImprovement: false,
						text: '소재 구분은 충분합니다. 빛 반사 포인트를 더 구조적으로 배치하면 제품성이 살아납니다.'
					}
				],
				INSTRUCTOR: [
					{
						axis: '구성력',
						status: '+5 필요',
						needImprovement: true,
						text: '문제 읽기 후 3분 내 썸네일 스케치를 끝내는 루틴을 고정하면 안정적입니다.'
					},
					{
						axis: '조형완성도',
						status: '+2 필요',
						needImprovement: true,
						text: '형태 정리에서 엣지를 조금 더 날카롭게 잡으면 산업디자인과 톤에 맞습니다.'
					}
				],
				PROFESSOR: [
					{
						axis: '주제해석',
						status: '합격권',
						needImprovement: false,
						text: '건국대는 발상 전개가 자연스럽기만 해도 강점이 됩니다. 그 흐름을 정리하는 문장화가 중요합니다.'
					},
					{
						axis: '구성력',
						status: '+5 필요',
						needImprovement: true,
						text: '메인 아이디어가 보이는 순간을 더 앞당겨야 합니다. 첫 인상에서 핵심 기능이 드러나야 합니다.'
					}
				]
			}
		},
		KOOKMIN: {
			key: 'KOOKMIN',
			label: '국민대',
			name: '국민대학교 시각디자인과',
			match: 78,
			rank: '상위 42%',
			stats: [
				{ label: '지원자', value: '1,180명' },
				{ label: '합격률', value: '7.9%' },
				{ label: '예상등수', value: '상위 42%', accent: true }
			],
			axisDistances: [
				{ label: '구성력', value: '+7 필요', needImprovement: true },
				{ label: '명암/질감', value: '+2 필요', needImprovement: true },
				{ label: '조형완성도', value: '합격권', needImprovement: false },
				{ label: '주제해석', value: '+4 필요', needImprovement: true }
			],
			strengths: ['채색 분위기가 눈에 잘 띔', '마감 완성도가 좋아 한 장의 밀도는 충분함'],
			improvements: ['국민대는 더 명확한 그래픽 구조를 선호함', '문제 해석이 직접적이라 서사 확장이 부족함'],
			historyNotes: [
				{
					label: '출제 경향 변화',
					tone: 'blue',
					text: '국민대는 메시지가 명확한 구성과 빠른 전달력을 선호하는 기조가 이어지고 있습니다.'
				},
				{
					label: '최근 변화',
					tone: 'gold',
					text: '평면 그래픽 감각과 타이포적 리듬을 함께 보는 채점 메모가 늘고 있습니다.'
				},
				{
					label: '핵심 인사이트',
					tone: 'rose',
					text: '감성적인 톤은 장점이지만, 정보를 더 빠르게 읽히도록 정리하는 훈련이 추가로 필요합니다.'
				}
			],
			historyRows: [
				{ year: '나의 점수', scores: [79, 74, 73, 83], diffs: [0, 0, 0, 0], isMine: true },
				{ year: '2024 커트', scores: [81, 76, 72, 87], diffs: [-2, -2, 1, -4] },
				{ year: '2023 커트', scores: [78, 74, 71, 84], diffs: [1, 0, 2, -1] },
				{ year: '2022 커트', scores: [76, 71, 69, 82], diffs: [3, 3, 4, 1] }
			],
			expertGuides: {
				AI: [
					{
						axis: '구성력',
						status: '+7 필요',
						needImprovement: true,
						text: '국민대는 메시지가 빠르게 읽혀야 합니다. 첫 3초 안에 중심 구조가 보이도록 블록을 재정렬하세요.'
					},
					{
						axis: '명암/질감',
						status: '+2 필요',
						needImprovement: true,
						text: '명암 차가 조금 더 강하면 그래픽 전달력이 살아납니다. 현재는 무드가 좋지만 대비가 약합니다.'
					}
				],
				INSTRUCTOR: [
					{
						axis: '주제해석',
						status: '+4 필요',
						needImprovement: true,
						text: '국민대는 주제를 정리한 한 문장을 화면에 그대로 적용하는 사고가 중요합니다. 문제문을 짧게 바꿔보세요.'
					},
					{
						axis: '조형완성도',
						status: '합격권',
						needImprovement: false,
						text: '마감은 이미 충분합니다. 표현을 덜어내도 강한 지점은 남게 만드는 편집이 필요합니다.'
					}
				],
				PROFESSOR: [
					{
						axis: '구성력',
						status: '+7 필요',
						needImprovement: true,
						text: '정보가 많은 화면보다 구조가 명쾌한 화면을 더 높게 평가합니다. 덜어내는 결단이 필요합니다.'
					},
					{
						axis: '주제해석',
						status: '+4 필요',
						needImprovement: true,
						text: '감성의 밀도는 좋지만, 메시지의 방향을 한 번 더 명시적으로 조형화해야 합니다.'
					}
				]
			}
		}
	}
};
