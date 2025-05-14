import { File } from "buffer";
import Image from "next/image";
import { useCallback } from "react";
import { useState } from "react";
import { motion } from "framer-motion";

export default function Result() {
    
     const download = () => {

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
                <Image src="/sheet.png" width={300} height={500} alt="sheet"/>
                <p className="mt-2">File name.pdf</p>
                <button className="bg-white text-black px-5 py-2 rounded-lg mt-4 cursor-pointer w-full flex justify-center hover:bg-[#DCDCDC]" onClick={() => download()}><Image src="/download.svg" width={25} height={25} alt="download"/>Telecharger</button>
           </motion.div>
        )
}