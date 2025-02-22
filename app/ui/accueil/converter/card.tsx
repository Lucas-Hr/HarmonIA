import Image from "next/image";
import { useCallback } from "react";
import { useState } from "react";


export default function Card() {
    const [file, setFile] = useState<File | null>(null);

    const handleDrop = useCallback((event : any) => {
        event.preventDefault();
        const droppedFile = event.dataTransfer.files[0];
        if (droppedFile) {
            setFile(droppedFile);
        }
    }, []);

    const handleFileChange = (event : any) => {
        const selectedFile = event.target.files[0];
        if (selectedFile) {
            setFile(selectedFile);
        }
    };
    
    return (
            <div className="flex flex-col py-8 items-center border-2 border-[#404040] bg-[#0D0D0D] w-[360px] h-auto rounded-lg" onDragOver={(e) => e.preventDefault()}
            onDrop={handleDrop}>
                <Image src="/upload.png" width={100} height={100} alt="upload"/>
                <p className="text-xl">Glissez votre fichier ici</p>
                <div className="mt-4 flex justify-between items-center">
                    <hr className="w-14 me-4"/>
                    <p className="text-2xl font-extralight">ou</p>
                    <hr className="w-14 ms-4"/>
                </div>
                {file && <p className="text-center mt-6">{file.name}</p>}
                <input type="file" id="fileInput" className="hidden"  onChange={handleFileChange}/>
                <label htmlFor="fileInput" className="bg-white text-black px-5 py-2 rounded-lg mt-4 cursor-pointer">
                    Parcourir
                </label>
                <button className="bg-[#000000] text-[#B6B6B6] px-5 py-2 rounded-lg mt-2 cursor-pointer border-2 border-[#404040]">Convertir</button>
            </div>
        )
}