import localHostEnv from "../config.js";
const buttonBackUserInfo=document.querySelector(".button--userInfo__back button");
buttonBackUserInfo.onclick=()=>{
    window.location="http://"+localHostEnv+"/admin/user";
}