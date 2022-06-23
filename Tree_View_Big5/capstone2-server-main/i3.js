async function asynchronous1() {
    console.log('Firsst 1');
    setTimeout(() => {
        console.log('asynchronous 1');
    }, 2000);
}
async function asynchronous2() {
    console.log('Firsst 2');

    setTimeout(() => {
        console.log('asynchronous 2');
    }, 2000);
}
async function asynchronous3() {
    console.log('Firsst 3');
    setTimeout(() => {
        console.log('asynchronous 3');
    }, 1999);
}

async function getABC() {
    // let A = await asynchronous1();
    // let B = await asynchronous2();
    // let C = await asynchronous3();
    await Promise.all([asynchronous1(), asynchronous2(), asynchronous3()]);

    // return A * B * C;
}

console.log(getABC());
console.log('ABC');
