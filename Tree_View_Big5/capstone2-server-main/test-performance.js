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

const searchTree = (children, id1, y = undefined) => {
    for (let child of children) {
        if (child.id.toString().trim() === id1.toString().trim()) {
            y = child;
        } else if (Array.isArray(child.children) && child.children.length) {
            y = searchTree(child.children, id1);
        }
    }
    if (y !== undefined) return y;
};

const convertRelated1 = (tempChildren) => {
    let res = [];
    tempChildren.forEach((child1) => {
        tempChildren.forEach((child2) => {
            const child = searchTree(res, child1.id);
            if (child1.id === child2.parentId) {
                if (child && child.children && child.children.length) {
                    child.children.push(child2);
                } else if (child && !child.children) {
                    child.children = [];
                    child.children.push(child2);
                } else {
                    const x = {
                        ...child1,
                        children: [child2],
                    };
                    res.push(x);
                }
            } else {
                if (!child) {
                    res.push(child1);
                }
            }
        });
    });
    return res;
};

const convertRelated2 = (tempChildren) => {
    let res = [];
    for (let i = 0; i < tempChildren.length - 1; i++) {
        for (let j = i + 1; j < tempChildren.length; j++) {
            const child = searchTree(res, tempChildren[i].id);

            if (tempChildren[i].id === tempChildren[j].parentId) {
                if (child && child.children && child.children.length) {
                    child.children.push(tempChildren[j]);
                } else if (child && !child.children) {
                    child.children = [];
                    child.children.push(tempChildren[j]);
                } else if (!child) {
                    const x = {
                        ...tempChildren[i],
                        children: [tempChildren[j]],
                    };
                    res.push(x);
                }
            } else {
                if (!child) {
                    res.push(tempChildren[i]);
                }
            }
        }
    }

    return res;
};

console.time('Function #1');
convertRelated1(children);
console.timeEnd('Function #1');

console.time('Function #2');
convertRelated2(children);
console.timeEnd('Function #2');
