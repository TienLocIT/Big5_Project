import toast from "./toast.js"
levels.forEach(level => {
    level.onchange = () => {
        var arrayValue = [];
        levels.forEach(data => {
            arrayValue.push(data.value);
        })
        // if(arrayValue.find(1)!=-1){
        //     toast({
        //             Title:"Error",
        //             Message:"The question just have one level high",
        //             Type:"error",
        //             Duration:2000
        //         })
        // }
        arrayValue = arrayValue.filter(element => (element != "2") && (element != "3"));
        if (arrayValue.length > 1) {
            toast({
                Title: "Error",
                Message: "The question just have one level high",
                Type: "error",
                Duration: 2000
            })
            arrayValue = [];
            level.value = 2;
        }
        return;
    }
})