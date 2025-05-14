import Image from "next/image"

export default function Card({text1 , text2} : {text1: string, text2: string}) {
    return (
        <div className="flex flex-col justify-between items-center border-2 border-[#404040] bg-[#0D0D0D] cursor-pointer w-[300px] h-[350px] rounded-lg py-8 hover:bg-[#272727]">
            <p className="text-4xl font-extralight">{text1}</p> 
            <Image src="/arrow.png" width={100} height={100} alt="arrow"/>
            <p className="text-4xl font-extralight">{text2}</p>
        </div>
    )
}