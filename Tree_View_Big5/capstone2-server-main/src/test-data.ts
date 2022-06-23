export interface TreeData {
    title: string;
    subtitle?: string;
    id?: string;
    children?: (TreeData | string)[];
    expanded?: boolean;
    [index: string]: string | boolean | undefined | (TreeData | string)[];
}

export const facet: TreeData[] = [
    {
        title: 'Openness',
        expanded: false,
        children: [
            { title: 'Ideas' },
            { title: 'Fantasy' },
            { title: 'Aesthetics' },
            { title: 'Actions' },
            { title: 'Feelings' },
            { title: 'Values' },
        ],
    },
    {
        title: 'Neuroticism',
        expanded: false,
        children: [
            { title: 'Anxiety' },
            { title: 'Angry Hostility' },
            { title: 'Depression' },
            { title: 'Self-consciousness' },
            { title: 'Impulsiveness' },
            { title: 'Vulnerability' },
        ],
    },
    {
        expanded: false,
        title: 'Conscientiousness',
        children: [
            { title: 'Competence' },
            { title: 'Order' },
            { title: 'Dutifulness' },
            { title: 'Achievement striving' },
            { title: 'Self-discipline' },
            { title: 'Deliberation' },
        ],
    },
    {
        expanded: false,
        title: 'Agreeableness',
        children: [
            { title: 'Trust' },
            { title: 'Straightforwardness' },
            { title: 'Altruism' },
            { title: 'Compliance' },
            { title: 'Modesty' },
            { title: 'Tender-mindedness' },
        ],
    },
    {
        expanded: false,
        title: 'Extraversion',
        children: [
            { title: 'Gregariousness' },
            { title: 'Assertiveness' },
            { title: 'Activity' },
            { title: 'Excitement-seeking' },
            { title: 'Positive emotions' },
            { title: 'Warmth' },
        ],
    },
];

export const keywords = [
    {
        title: 'food',
        expanded: false,
        children: [
            {
                title: 'food material',
                expanded: false,
                children: [
                    {
                        title: 'food product',
                        expanded: false,
                        children: [
                            {
                                title: 'food product by meal type',
                                expanded: false,
                            },
                        ],
                    },
                    {
                        title: 'animal feed',
                        expanded: false,
                        children: [
                            {
                                title: 'contaminated animal feed',
                                expanded: false,
                            },
                        ],
                    },
                    {
                        title: 'plant feed',
                        expanded: false,
                        children: [
                            {
                                title: 'bone meal',
                                expanded: false,
                            },
                        ],
                    },
                ],
            },
        ],
    },
    {
        title: 'occurrent',
        expanded: false,
        children: [
            {
                title: 'continuant',
                expanded: false,
                children: [
                    {
                        title: 'independent continuant',
                        expanded: false,
                        children: [
                            {
                                title: 'chemical entity',
                                expanded: false,
                                children: [
                                    {
                                        title: 'molecular entity',
                                        expanded: false,
                                        children: [
                                            {
                                                title: 'inorganic molecular entity',
                                                expanded: false,
                                                children: [
                                                    {
                                                        title: 'inorganic hydride',
                                                        expanded: false,
                                                        children: [
                                                            {
                                                                title: 'chalcogen hydride',
                                                                expanded: false,
                                                                children: [
                                                                    {
                                                                        title: 'oxygen hydride',
                                                                        expanded:
                                                                            false,
                                                                        children:
                                                                            [
                                                                                {
                                                                                    title: 'water',
                                                                                    expanded:
                                                                                        false,
                                                                                },
                                                                            ],
                                                                    },
                                                                ],
                                                            },
                                                        ],
                                                    },
                                                ],
                                            },
                                        ],
                                    },
                                ],
                            },
                            {
                                title: 'generically dependent continuant',
                                expanded: false,
                                children: [
                                    {
                                        title: 'information content entity',
                                        expanded: false,
                                    },
                                ],
                            },
                        ],
                    },
                    {
                        title: 'independent continuant',
                        expanded: false,
                        children: [
                            {
                                title: 'anatomical entity',
                                expanded: false,
                            },
                        ],
                    },
                    {
                        title: 'specifically dependent continuant',
                        expanded: false,
                        children: [
                            {
                                title: 'quality',
                                expanded: false,
                            },
                        ],
                    },
                ],
            },
            {
                title: 'process',
                expanded: false,
                children: [
                    {
                        title: 'planned process',
                        expanded: false,
                        children: [
                            {
                                title: 'well intervention',
                                expanded: false,
                            },
                        ],
                    },
                    {
                        title: 'environmental system process',
                        expanded: false,
                        children: [
                            {
                                title: 'adiabatic process',
                                expanded: false,
                            },
                            {
                                title: 'storm',
                                expanded: false,
                            },
                        ],
                    },
                    {
                        title: 'biological process',
                        expanded: false,
                        children: [
                            {
                                title: 'behavior',
                                expanded: false,
                            },
                        ],
                    },
                ],
            },
        ],
    },
];
