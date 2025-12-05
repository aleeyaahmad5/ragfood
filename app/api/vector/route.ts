import { Index } from "@upstash/vector";
import { NextResponse } from "next/server";

const index = new Index({
  url: process.env.UPSTASH_VECTOR_REST_URL,
  token: process.env.UPSTASH_VECTOR_REST_TOKEN,
})

export const GET = async () => {
  const result = await index.fetch(["vector-id"], { includeData: true })
  
  return new NextResponse(
    JSON.stringify({ result: result[0] }),
    { status: 200 }
  )
}