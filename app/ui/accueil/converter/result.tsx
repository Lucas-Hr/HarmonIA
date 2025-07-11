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
     midiFile64 : any | null,
     audioUrl : string | null,
     spectrogramURL : string | null,
     abcNotation : string | null,
     xmlFile : string | null,
     xmlFile64 : any | null,
}

export default function Result({file, textOne, image, midiFile, midiFile64, audioUrl, spectrogramURL, abcNotation, xmlFile,xmlFile64} : ResultProps) {
     useEffect(() => {
         
}    ,[])

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
          // downloadFile(file3, name3);
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

     const downloadMidiAndImage = () => {
          if (midiFile64 && image && xmlFile64) {
            const midiUrl = base64ToUrl(midiFile64, 'audio/midi');
            const xmlUrl = base64ToUrl(midiFile64, 'audio/xml');
            

            // Télécharger les deux fichiers
            downloadBoth(midiUrl, `${file.name.replace(".wav","_midi.midi")}`, image, `${file.name.replace(".wav","_partition.png")}`);
            
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
               {image && midiFile &&(
                    <>

                         <Image src={image} width={200} height={200} alt="piano rolls" className="bg-white"/>
                         <p className="mt-2">{file.name.replace('.wav', '_partition.png')}</p>
                         <button className="bg-white text-black px-5 py-2 rounded-lg mt-4 cursor-pointer w-full flex justify-center hover:bg-[#DCDCDC]" onClick={downloadMidiAndImage}><Image src="/download.svg" width={25} height={25} alt="download"/>Telecharger</button>
                    </>
               )}
      
               {audioUrl && spectrogramURL && (
                    <>
                         <audio controls className="mt-4">
                              <source src={audioUrl} />
                         </audio>
                         <p className="mt-2 txt-center">{file.name.replace('.midi', '_audio.mp3')}</p>
                         <Image src={spectrogramURL} width={200} height={200} alt="spectrogram" className="mt-2"/>
                         <p className="mt-2 txt-center">{file.name.replace('.midi', '_spectrogram.png')}</p>
                         <button className="bg-white text-black px-5 py-2 rounded-lg mt-4 cursor-pointer w-full flex justify-center hover:bg-[#DCDCDC]" onClick={() => downloadBoth(audioUrl, `${file.name.replace('.midi', '_audio.wav')}`, spectrogramURL, `${file.name.replace('.midi', '_spectrogram.png')}`)}><Image src="/download.svg" width={25} height={25} alt="download"/>Telecharger</button>
                    </>   
               )}
                
           </motion.div>
        )
}