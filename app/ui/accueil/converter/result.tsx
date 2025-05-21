// import { File } from "buffer";
import Image from "next/image";
import { useCallback, useEffect } from "react";
import { useState } from "react";
import { motion } from "framer-motion";

type ResultProps = {
     file : any;
     textOne : string | null,
     image : string | null
}

export default function Result({file, textOne, image} : ResultProps) {
     const [audioUrl, setAudioUrl] = useState<string | null>("");
     const [pdfUrl, setPdfUrl] = useState<string | null>("");
     // useEffect(() => {
     //      const url = URL.createObjectURL(file);
     //      if (textOne === "Partition"){
     //           setPdfUrl(url);
     //      } else setAudioUrl(url)
     // },[])

     const download = (f : any) => {
          if (!f) return;
          // const url = URL.createObjectURL(f); // Create a blob URL for the File   
          const link = document.createElement("a");
          link.href = f;
          link.download = f.name; // Suggested filename
          document.body.appendChild(link);
          link.click();
          document.body.removeChild(link);
          // URL.revokeObjectURL(url); // Free memory
          // console.log(audioUrl)          
     }

    return (
           <motion.div 
           className="flex flex-col items-center ms-4"
           initial={{opacity:0 , x:-20}}
           animate={{opacity:1, x:0}}
           transition={{
            duration : 0.5
           }}
           >
               {image && <Image src={image} width={200} height={200} alt="piano rolls"/>}
      
               {file?.type === ('audio/mpeg') && audioUrl && (
                    <audio controls src={audioUrl} className="w-full"></audio>
               )}
                <p className="mt-2">{file.name}</p>
                <button className="bg-white text-black px-5 py-2 rounded-lg mt-4 cursor-pointer w-full flex justify-center hover:bg-[#DCDCDC]" onClick={() => download(image)}><Image src="/download.svg" width={25} height={25} alt="download"/>Telecharger</button>
           </motion.div>
        )
}