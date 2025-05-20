'use client'
import React from 'react'
import CardTestimonial from './cardTestimonial'
import { Swiper, SwiperSlide } from 'swiper/react'
import {Pagination} from 'swiper/modules'
import { motion } from 'framer-motion'
import 'swiper/css';
import 'swiper/css/pagination';

export default function testimonial() {
  const clients = [
    {
      name : 'Kevin Smith',
      job : 'Guitariste',
      text : 'Lorem ipsum dolor sit amet consectetur adipisicing elit. Odio similique quia nostrum! Vel nemo illum placeat possimus excepturi modi. Voluptatem magnam mollitia pariatur impedit sint quibusdam quos recusandae adipisci iure.',
      image : '/firstOne.jpg'
    },
    {
      name : 'Allan Jesus',
      job : 'Pianiste',
      text : 'Lorem ipsum dolor sit amet consectetur adipisicing elit. Odio similique quia nostrum! Vel nemo illum placeat possimus excepturi modi. Voluptatem magnam mollitia pariatur impedit sint quibusdam quos recusandae adipisci iure.',
      image : '/secondOne.jpg'
    },
    {
      name : 'Martinez James',
      job : 'Musicien',
      text : 'Lorem ipsum dolor sit amet consectetur adipisicing elit. Odio similique quia nostrum! Vel nemo illum placeat possimus excepturi modi. Voluptatem magnam mollitia pariatur impedit sint quibusdam quos recusandae adipisci iure.',
      image : '/thirdOne.jpg'
    },
    {
      name : 'Synyster Gates',
      job : 'Batteur',
      text : 'Lorem ipsum dolor sit amet consectetur adipisicing elit. Odio similique quia nostrum! Vel nemo illum placeat possimus excepturi modi. Voluptatem magnam mollitia pariatur impedit sint quibusdam quos recusandae adipisci iure.',
      image : '/fourthOne.jpg'
    },
  ]
  return (

    <div className='px-40  py-40 w-screen '>
      <motion.h1
        initial={{opacity : 0, y : 50}}
        whileInView={{opacity : 1, y : 0}}
        transition={{
            duration : 1,
            delay : 0.25
        }}
        viewport={{once : true}}
        className="text-white text-5xl text-center font-lighter mb-10">Testimonials
        </motion.h1>
        <Swiper
        modules={[Pagination]}
        spaceBetween={150}
        slidesPerView={3} 
        // pagination={{ clickable: true }}
        className='z-1 cursor-grab'
        >
            {clients.map((client,index) => {
              return(
                <SwiperSlide key={index}>
                    <CardTestimonial name={client.name} job={client.job} text={client.text} image={client.image} />
                </SwiperSlide>
              )
            })}
    </Swiper>
    </div>
    
  )
}
