'use client'
import Title from "@/app/ui/accueil/converter/title"
import Card from "@/app/ui/accueil/converter/card"
import { useSearchParams } from "next/navigation"
export default function Page() {
    const searchParams = useSearchParams()
    const textOne = searchParams.get('textOne')
    const textTwo = searchParams.get('textTwo')
    console.log(textOne)
    return (
        <div className="px-40 h-screen pt-32">
            <Title text1={textOne || ''} text2={textTwo || ''}/>
            <div className="flex justify-center items-center mt-10">
                <Card />
            </div>            
        </div>
    )
}