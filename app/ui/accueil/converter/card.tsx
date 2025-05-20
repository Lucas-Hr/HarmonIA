import { File } from "buffer";
import Image from "next/image";
import { useCallback } from "react";
import { useState } from "react";
import { motion } from "framer-motion";

type CardProps = {
    setIsConverted: (value: boolean) => void,
    file : File | null,
    setFile : (value : File | null) => void,
    textOne : string | null
};

export default function Card({setIsConverted , file, setFile, textOne}: CardProps) {

    const handleDrop = useCallback((event : any) => {
        event.preventDefault();
        const droppedFile = event.dataTransfer.files[0];
        if (droppedFile && textOne === "Partition") {
            if (droppedFile.type.startsWith('application/pdf')){
                setFile(droppedFile);
            } else alert("File type unsupported")
            
        } else if (droppedFile && textOne === "Musique") {
            if (droppedFile.type.startsWith('audio/mpeg')){
                setFile(droppedFile);
            } else alert("File type unsupported")
            
        }
    }, []);

    const handleFileChange = (event : any) => {
        const selectedFile = event.target.files[0];
        if (selectedFile && textOne === "Partition") {
            if (selectedFile.type.startsWith('application/pdf')){
                setFile(selectedFile);
            } else alert("File type unsupported")
            
        } else if (selectedFile && textOne === "Musique") {
            if (selectedFile.type.startsWith('audio/mpeg')){
                setFile(selectedFile);
            } else alert("File type unsupported")
            
        }
    };
    

    const [isVisible, setIsVisible] = useState(false)

    const converting = (f : File | null) => {
        if(f === null){
            alert("Please Insert a file")
        } else {
       setIsVisible(true);
       setTimeout(() => {
            setIsVisible(false)
            setIsConverted(true)
       },5000)   
        }
    }


    return (
            <motion.div 
            initial={{y : 50,opacity : 0}}
            animate={{y : 0, opacity : 1}}
            transition={{
                duration : 1,
                delay : 0.5
            }}
            className="flex flex-col py-8 items-center border-2 border-[#404040] bg-[#0D0D0D] w-[360px] h-auto rounded-lg relative" 
            onDragOver={(e) => e.preventDefault()}
            onDrop={handleDrop}>
                {isVisible && 
                <motion.div 
                initial={{opacity : 0}}
                animate={{opacity : 1}}
                transition={{
                    duration : 0.5
                }}
                className="flex flex-col justify-center center absolute border-[#404040] bg-[black] w-[360px] h-full rounded-lg top-0">
                    <div className="flex flex-col items-center justify-center">
                        <motion.h1 
                        animate={{
                            y: [0, -5, 0]
                        }}
                        transition={{
                            repeat : Infinity,
                            repeatType : "mirror",
                            duration : 1,
                            ease : "easeInOut"
                        }}
                        className="font-semibold text-lg">Converting ...</motion.h1>
                        <Image src="/converting.gif" width={200} height={200} alt="convert"/>
                    </div>
                    
                </motion.div>}
                <Image src="/upload.png" width={100} height={100} alt="upload"/>
                <p className="text-xl">Glissez votre fichier ici</p>
                <div className="mt-4 flex justify-between items-center">
                    <hr className="w-14 me-4"/>
                    <p className="text-2xl font-extralight">ou</p>
                    <hr className="w-14 ms-4"/>
                </div>
                {file && <p className="text-center mt-6">{file.name}</p>}
                <input type="file" id="fileInput" className="hidden"  onChange={handleFileChange}/>
                <label htmlFor="fileInput" className="bg-white text-black px-5 py-2 rounded-lg mt-4 cursor-pointer hover:bg-[#DCDCDC]">
                    Parcourir
                </label>
                <button className="bg-[#000000] text-[#B6B6B6] px-5 py-2 rounded-lg mt-2 cursor-pointer border-2 border-[#404040] hover:bg-[#171717]" onClick={() => converting(file)}>Convertir</button>
            </motion.div>
        )
}