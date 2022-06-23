import localHostEnv from "../config.js";
import fetchAPI from "../fetchAPI.js";
const tbody=document.querySelector(".tableFixHead table tbody");
const table=document.querySelector(".tableFixHead table");
const loadingio=document.getElementById("loadingio");
const traitOption=document.getElementById("trait__Option");
const addressModal = document.querySelector(".address-modal");
const loadingioModal = document.querySelector(".loadingiomodal");
const addressBody = document.querySelector(".address__body");
const addressModalButtonCancel = document.querySelector(".address-modal__button-cancel");
console.log(addressModal)

async function getResponse(url){
    const response=await fetchAPI(url,{},"GET");
    return response;
} 
window.onload=async()=>{
    loadingio.style.display="block";
    table.style.display="none";
    const datas=await getResponse("http://"+localHostEnv+"/admin/getusers?trait=All");
    const datasFilter = datas.filter(data => data.isClone === false);
    console.log(datasFilter);
    datasFilter.forEach(data=>{
        tbody.innerHTML=tbody.innerHTML+`
         <tr>
                <td class="nameUser--table"><div class="buttonInfo" id=${data._id}>${data.userName}</div></td>
                <td>${data.arrayPersonality[0].point}</td>
                <td>${data.arrayPersonality[1].point}</td>
                <td>${data.arrayPersonality[2].point}</td>
                <td>${data.arrayPersonality[3].point}</td>
                <td>${data.arrayPersonality[4].point}</td>
                <td><button class="view_address" idUser=${data._id}>View address</button><td>
              </tr>
        `
    })
    loadingio.style.display="none";
    table.style.display="table";
    const buttonInfo=[...document.getElementsByClassName("buttonInfo")];
    const viewAddressButton = [...document.getElementsByClassName("view_address")];
    buttonInfo.forEach(data=>{
        data.onclick=()=>{
            loadingio.style.display="block";
            table.style.display="none";
            window.location="http://"+localHostEnv+"/admin/user?id="+data.getAttribute('id');
        }
    })
    viewAddressButton.forEach(data => {
        data.onclick = async () => {
            
            addressModal.style.display = "flex";
            loadingioModal.style.display = "block";
            const url = "http://"+localHostEnv+"/admin/getAddress?id="+data.getAttribute('idUser');
            const dataAdress =await fetchAPI(url,{},"GET");
            if(dataAdress.length == 0){
                addressBody.innerHTML = `<div style="text-align:center">User don't have location yet </div>`
            }
            dataAdress.forEach((data,index) =>{
                addressBody.innerHTML +=`<div style="text-align:center;margin:2% 0;">${data}</div>` 
            })
            loadingioModal.style.display = "none";
        }
    })
    addressModalButtonCancel.onclick = () => {
        addressBody.innerHTML ="";
        addressModal.style.display ="none";
    }
    
}
traitOption.onchange=async()=>{
    const buttonInfo=[...document.getElementsByClassName("buttonInfo")];
    tbody.innerHTML="";
    loadingio.style.display="block";
    table.style.display="none";
    const traitString=traitOption.options[traitOption.selectedIndex].value;
    const datas=await getResponse("http://"+localHostEnv+"/admin/getusers?trait="+traitString);
    const datasFilter = datas.filter(data => data.isClone === false);
    datasFilter.forEach(data=>{
        tbody.innerHTML=tbody.innerHTML+`
         <tr>
                <td class="nameUser--table"><span class="buttonInfo" name=${data.userName}>${data.userName}</span></td>
                <td>${data.arrayPersonality[0].point}</td>
                <td>${data.arrayPersonality[1].point}</td>
                <td>${data.arrayPersonality[2].point}</td>
                <td>${data.arrayPersonality[3].point}</td>
                <td>${data.arrayPersonality[4].point}</td>
              </tr>
        `
    })
    loadingio.style.display="none";
    table.style.display="table";
    buttonInfo.forEach(data=>{
        data.onclick=()=>{
            table.style.display="none";
            loadingio.style.display="block";
            window.location="http://"+localHostEnv+"/admin/user?id="+data.getAttribute('id');
        }
    })
}
