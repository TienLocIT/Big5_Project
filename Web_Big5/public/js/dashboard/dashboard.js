 import localHostEnv from "../config.js";
 import fetchAPI from "../fetchAPI.js";
const container=document.querySelector(".container");
const loadingio=document.getElementById("loadingiodashboard");
const mainLeft=document.querySelector(".main__left");
const mainRight=document.querySelector(".main__right");
async function getData(url){
    container.style.gridTemplateColumns="18rem auto";
    loadingio.style.display="block";
    const response=await fetchAPI(url,{},"GET");
    return response;
}
getData("http://"+localHostEnv+"/admin/numbertrait").then(data=>{
    if(data==null){
        console.log(null)
    }
    else{
        const ctx = document.getElementById('myChart').getContext('2d');
        const ctxQuestion=document.getElementById('myChartQuestion').getContext('2d');
        const myChart=new Chart(ctx, {
            type: 'bar',
            data: {
                labels: ['Openness', 'Conscientious', 'Extraversion', 'Agreeable', 'Neuroticism'],
                datasets: [{
                    data: data.arrayUser,
                    backgroundColor: [
                        'rgba(255, 99, 132, 0.2)',
                        'rgba(54, 162, 235, 0.2)',
                        'rgba(255, 206, 86, 0.2)',
                        'rgba(75, 192, 192, 0.2)',
                        'rgba(153, 102, 255, 0.2)',
                        
                    ],
                    borderColor: [
                        'rgba(255, 99, 132, 1)',
                        'rgba(54, 162, 235, 1)',
                        'rgba(255, 206, 86, 1)',
                        'rgba(75, 192, 192, 1)',
                        'rgba(153, 102, 255, 1)',
                    ],
                    borderWidth: 1
                }]
            },
            options: {
                title: {
                    display: true,
                    text: 'Number trait user'
                },
                title: {
                    display: true,
                    text: 'Number trait user'
                },
                plugins: {
                    legend: {
                      display: false
                    },
                  },
            
                scales: {
                    y: {
                        beginAtZero: true
                    }
                }
            }
        });
        const myChartQuestion=new Chart(ctxQuestion, {
            type: 'bar',
            data: {
                labels: ['Openness', 'Conscientious', 'Extraversion', 'Agreeable', 'Neuroticism'],
                datasets: [{
                    data: data.arrayQuestion,
                    backgroundColor: [
                        'rgba(255, 99, 132, 0.2)',
                        'rgba(54, 162, 235, 0.2)',
                        'rgba(255, 206, 86, 0.2)',
                        'rgba(75, 192, 192, 0.2)',
                        'rgba(153, 102, 255, 0.2)',
                        
                    ],
                    borderColor: [
                        'rgba(255, 99, 132, 1)',
                        'rgba(54, 162, 235, 1)',
                        'rgba(255, 206, 86, 1)',
                        'rgba(75, 192, 192, 1)',
                        'rgba(153, 102, 255, 1)',
                    ],
                    borderWidth: 1
                }]
            },
            options: {
                plugins: {
                    title: {
                        display: true,
                        text: 'Number trait question'
                    },
                    legend: {
                      display: false
                    },
                  },
            
                scales: {
                    y: {
                        beginAtZero: true
                    }
                }
            }
        });
    }
    container.style.gridTemplateColumns="18rem auto 29rem";
    mainLeft.style.display="block";
    mainRight.style.display="flex";
    loadingio.style.display="none";
})