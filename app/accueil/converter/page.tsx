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
    const searchParams = useSearchParams()
    const textOne = searchParams.get('textOne')
    const textTwo = searchParams.get('textTwo')
    return (
        <div
        className="px-40 h-screen pt-32">
            <Title text1={textOne || ''} text2={textTwo || ''}/>
            <div className="flex justify-center items-center mt-10">
                <Card setIsConverted={setIsConverted} file ={file} setFile={setFile}/>
                {isConverted && <Result file={file}/>}
            </div>            
        </div>
    )
}