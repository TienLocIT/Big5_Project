function waitUntil(fn,condition,interval){
    interval = interval || 100;
    var shell=function(){
        var timer=setInterval(
            function(){
                var check;
                try{
                    check=condition()
                }
                catch{
                    check=false;
                }
                if(check){
                    clearInterval(timer);
                    delete timer;
                    fn();
                }
            }
        ),
        interval
    }
    return shell;
}