// import { File } from "buffer";
import Image from "next/image";
import { useCallback, useEffect } from "react";
import { useState } from "react";
import { motion } from "framer-motion";

type ResultProps = {
     file : any;
     textOne : string | null,
     image : string | null,
     midiFile : string | null,
     midiFile64 : any | null
}

export default function Result({file, textOne, image, midiFile, midiFile64} : ResultProps) {
     const [audioUrl, setAudioUrl] = useState<string | null>("");
     const [pdfUrl, setPdfUrl] = useState<string | null>("");
     // useEffect(() => {
     //      const url = URL.createObjectURL(file);
     //      if (textOne === "Partition"){
     //           setPdfUrl(url);
     //      } else setAudioUrl(url)
     // },[])

     // const download = (f : any) => {
     //      if (!f) return;
     //      // const url = URL.createObjectURL(f); // Create a blob URL for the File   
     //      const link = document.createElement("a");
     //      link.href = f;
     //      link.download = f.name; // Suggested filename
     //      document.body.appendChild(link);
     //      link.click();
     //      document.body.removeChild(link);
     //      // URL.revokeObjectURL(url); // Free memory
     //      // console.log(audioUrl)          
     // }

     const downloadBoth = (file1: string, name1: string, file2: string, name2: string) => {
          if (!file1 || !file2) return;
      
          const downloadFile = (url: string, name: string) => {
              const link = document.createElement("a");
              link.href = url;
              link.download = name;
              document.body.appendChild(link);
              link.click();
              document.body.removeChild(link);
          };
      
          downloadFile(file1, name1);
          downloadFile(file2, name2);
      };

      const base64ToUrl = (base64Data : any, mimeType : any) => {
          const byteCharacters = atob(base64Data);
          const byteNumbers = new Array(byteCharacters.length);
          
          for (let i = 0; i < byteCharacters.length; i++) {
            byteNumbers[i] = byteCharacters.charCodeAt(i);
          }
          
          const byteArray = new Uint8Array(byteNumbers);
          const blob = new Blob([byteArray], { type: mimeType });
          return URL.createObjectURL(blob);
        };

       // Fonction pour télécharger le fichier MIDI
     const downloadMidi = () => {
          if (midiFile64 && midiFile) {
          const midiUrl = base64ToUrl(midiFile64, 'audio/midi');
          
          const link = document.createElement('a');
          link.href = midiUrl;
          link.download = midiFile;
          document.body.appendChild(link);
          link.click();
          document.body.removeChild(link);
          URL.revokeObjectURL(midiUrl);
          }
     };

     const downloadMidiAndImage = () => {
          if (midiFile64 && image) {
            const midiUrl = base64ToUrl(midiFile64, 'audio/midi');

            // Télécharger les deux fichiers
            downloadBoth(midiUrl, 'midi version.mid', image, 'piano roll.png');
            
            // Nettoyer l'URL du MIDI
            setTimeout(() => URL.revokeObjectURL(midiUrl), 100);
          }
        };
 
     

    return (
           <motion.div 
           className="flex flex-col items-center ms-4"
           initial={{opacity:0 , x:-20}}
           animate={{opacity:1, x:0}}
           transition={{
            duration : 0.5
           }}
           >
               {image && midiFile && (
                    <>
                         <Image src={image} width={200} height={200} alt="piano rolls"/>
                         <p className="mt-2">{file.name}</p>
                         <button className="bg-white text-black px-5 py-2 rounded-lg mt-4 cursor-pointer w-full flex justify-center hover:bg-[#DCDCDC]" onClick={downloadMidiAndImage}><Image src="/download.svg" width={25} height={25} alt="download"/>Telecharger</button>
                    </>
                    )}
      
               {/* {file?.type === ('audio/mpeg') && audioUrl && (
                    <audio controls src={audioUrl} className="w-full"></audio>
               )} */}
                
           </motion.div>
        )
}