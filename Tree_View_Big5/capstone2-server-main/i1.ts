// const X = function () {
//     this.name = 1;
// };
// console.log(new X());

// y = 1;
// var y;
// console.log(y);

// const x: any = [{ a: [1, 2, 3] }];
// let f = x.find((x) => true);
// f.a.push(4);
// console.log(x);

const children = [
    {
        title: 'Food Fruit',
        id: 'PPyhSeUIsma4uMJa0XYHz',
        childId: undefined,
        parentId: 'qbXhyKtp1fuuGPKBTbvjl',
        parentTitle: 'Food',
    },
    {
        title: 'Food Meat',
        id: 'vr3VDIZR9HJFLOsMmPd7k',
        childId: undefined,
        parentId: 'qbXhyKtp1fuuGPKBTbvjl',
        parentTitle: 'Food',
    },
    {
        title: 'Smoke',
        id: 'w-toU3zUIPZqV66afyREn',
        childId: undefined,
        parentId: 'qbXhyKtp1fuuGPKBTbvjl',
        parentTitle: 'Food',
    },
    {
        title: 'Rice Field',
        id: 'PCx_HQXZjzBYE3713nbsD',
        childId: undefined,
        parentId: 'qbXhyKtp1fuuGPKBTbvjl',
        parentTitle: 'Food',
    },
    {
        title: 'Food Salack',
        id: '-Sj8ddRYJCpo5TcDS0Lcp',
        childId: undefined,
        parentId: 'qbXhyKtp1fuuGPKBTbvjl',
        parentTitle: 'Food',
        children: [
            {
                title: 'Food Salack 1',
                id: 'xbmpD-pABxJxJFx5NQNqn',
                childId: undefined,
                parentId: '-Sj8ddRYJCpo5TcDS0Lcp',
                parentTitle: 'Food Salack',
                children: [
                    {
                        title: 'Food Salack 1 1',
                        id: 'n5_ElQdim-SbRIV0A3-OO',
                        childId: undefined,
                        parentId: 'xbmpD-pABxJxJFx5NQNqn',
                        parentTitle: 'Food Salack 1',
                    },
                    {
                        title: 'Food Salack 1 2',
                        id: 'n5_ElQdim-SbRIV0A3-O1',
                        childId: undefined,
                        parentId: 'xbmpD-pABxJxJFx5NQNqn',
                        parentTitle: 'Food Salack 1',
                    },
                ],
            },
            {
                title: 'Food Salack 2',
                id: 'Mwoe6Dgd8XgPocpxxXmOu',
                childId: undefined,
                parentId: '-Sj8ddRYJCpo5TcDS0Lcp',
                parentTitle: 'Food Salack',
            },
            {
                title: 'Food Salack 3',
                id: 'cFcl8H5Wt6-nhA4cZojqn',
                childId: undefined,
                parentId: '-Sj8ddRYJCpo5TcDS0Lcp',
                parentTitle: 'Food Salack',
            },
        ],
    },
];

export const searchTree = (children, id1, id2, y = undefined) => {
    for (let child of children) {
        if (
            child.id.toString().trim() === id1.toString().trim() ||
            child.id === id2
        ) {
            y = child;
        } else if (Array.isArray(child.children) && child.children.length) {
            y = searchTree(child.children, id1, id2);
        }
    }
    if (y !== undefined) return y;
};

console.log(
    searchTree(children, 'n5_ElQdim-SbRIV0A3-OO1', 'n5_ElQdim-SbRIV0A3-OO1'),
);
