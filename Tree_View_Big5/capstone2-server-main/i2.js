let b = Array.from({ length: 1000 });

const f1 = (arr) => {
    let x;
    for (let i = 0; i < arr; i++) {
        x += i;
    }
    return x;
};

const f2 = (arr) => {
    let x;
    // arr.map((y) => {
    //     setTimeout(() => {
    //         console.log(y);
    //     }, 1000);
    //     x += y;
    // });
    for (let y of arr) {
        x += y;
    }
    return x;
};

console.time('Function #1');
f1(b);
console.timeEnd('Function #1');

console.time('Function #2');
f2(b);
console.timeEnd('Function #2');
