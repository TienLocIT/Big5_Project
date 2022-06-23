import localHostEnv from "./config.js";
const logoutButton=document.getElementById("logout__button");
logoutButton.onclick=()=>{
    const response=await fetch("http://"+localHostEnv+"/admin/logout",{
        method:"POST",
        mode:"cors",
        cache:"no-cache",
        credentials:"same-origin",
        headers:{
                "Content-type":"application/json"
        },
        redirect:"follow",
        referrerPolicy:"no-referrer",
        body:JSON.stringify(data)
    });
    const dataResponse=await response.json();
    if(dataResponse.error="No"){
        window.location="http://"+localHostEnv+"/login"
    }
}