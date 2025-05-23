'use client'
import React from 'react'
import Image from 'next/image'

type testimonialProps = {
    name : String,
    job : String,
    text : String,
    image : any
}

export default function cardTestimonial({name, job, text, image} : testimonialProps) {
  return (
    <div className='pt-16 pb-8 px-10 text-center border rounded-md relative z-1 bg-[black] hover:bg-[#272727] cursor-pointer'>
        <Image className='absolute top-[-50px] left-20 rounded-[100%] z-2 ' src={image} width={150} height={100} alt={image}/>
        <h1 className='font-bold text-xl'>{name}</h1>
        <h5 className='font-semibold text-[#B6B6B6] text-md mt-1'>{job}</h5>
        <p className='text-sm text-[#B6B6B6] text-base/6 mt-4'>"{text}"</p>
    </div>
  )
}
