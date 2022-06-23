import {
    Controller,
    Get,
    Post,
    Body,
    Patch,
    Param,
    Delete,
    Query,
    Put,
} from '@nestjs/common';
import { QuestionsService } from './questions.service';
import { CreateQuestionDto } from './dto/create-question.dto';
import { UpdateQuestionDto } from './dto/update-question.dto';
import { GetSuggestQuestionDto } from '@modules/suggest_questions/dto/get-suggest_question.dto';

@Controller('questions')
export class QuestionsController {
    constructor(private readonly questionsService: QuestionsService) {}

    @Post('get-output-model-result')
    async apiGetOutputModelResult(@Body() createQuestionDto: any) {
        return await this.questionsService.apiGetOutputModelResult(
            createQuestionDto,
        );
    }

    @Put()
    updateQuestion(@Body() createQuestionDto: CreateQuestionDto) {
        return this.questionsService.updateQuestion(createQuestionDto);
    }

    @Get()
    findSomeByKeywordAndFacet(@Query() payload: GetSuggestQuestionDto) {
        return this.questionsService.findSomeByKeywordAndFacet(payload);
    }

    @Patch(':id')
    update(
        @Param('id') id: string,
        @Body() updateQuestionDto: UpdateQuestionDto,
    ) {
        // return this.questionsService.update(+id, updateQuestionDto);
    }

    @Delete(':id')
    remove(@Param('id') id: string) {
        return this.questionsService.remove(+id);
    }
}
