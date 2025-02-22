import Image from "next/image";

export default function Card() {

    const dropHandler = (e : any) : void => {
        e.preventDefault();
        const file = e.dataTransfer.files[0];
        console.log("dropped : ",file);
        
    }


    return (
            <div className="flex flex-col pt-16 items-center border-2 border-[#404040] bg-[#0D0D0D] cursor-pointer w-[300px] h-[350px] rounded-lg" onDrop={dropHandler}>
                <Image src="/upload.png" width={100} height={100} alt="upload"/>
                <p className="text-xl">Glissez votre fichier ici</p>
                <div className="mt-4 flex justify-between items-center">
                    <hr className="w-14 me-4"/>
                    <p className="text-2xl font-extralight">ou</p>
                    <hr className="w-14 ms-4"/>
                </div>
                <input type="file" id="fileInput" className="hidden" />
                <label htmlFor="fileInput" className="bg-white text-black px-5 py-2 rounded-lg mt-6 cursor-pointer">
                    Parcourir
                </label>
            </div>
        )
}