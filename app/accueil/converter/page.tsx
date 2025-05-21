'use client'
import Title from "@/app/ui/accueil/converter/title"
import Card from "@/app/ui/accueil/converter/card"
import Result from "@/app/ui/accueil/converter/result"
import { useSearchParams } from "next/navigation"
import { useState } from "react"
import { File } from "buffer"
export default function Page() {
    const [isConverted, setIsConverted] = useState(false);
    const [file, setFile] = useState<File | null>(null);
    const [image, setImage] = useState<string | null>("")
    const [midiFile, setMidiFile] = useState<string | null>("")
    const [midiFile64, setMidiFile64] = useState<any | null>("")
    const [audioUrl, setAudioUrl] = useState<string | null>("")
    const [spectrogramURL, setSpectrogramUrl] = useState<string | null>("")
    const searchParams = useSearchParams()
    const textOne = searchParams.get('textOne')
    const textTwo = searchParams.get('textTwo')
    return (
        <div
        className="px-40 h-screen pt-20">
            <Title text1={textOne || ''} text2={textTwo || ''}/>
            <div className="flex justify-center items-center mt-10">
                <Card setIsConverted={setIsConverted} file ={file} 
                setFile={setFile} textOne={textOne} setImage={setImage}
                 setMidiFile={setMidiFile} setMidiFile64={setMidiFile64}
                 setAudioUrl={setAudioUrl} setSpectrogramUrl={setSpectrogramUrl}
                 />
                {isConverted && <Result file={file} textOne={textOne} image={image} midiFile={midiFile} midiFile64={midiFile64}
                    audioUrl={audioUrl} spectrogramURL={spectrogramURL}
                />}
            </div>            
        </div>
    )
}